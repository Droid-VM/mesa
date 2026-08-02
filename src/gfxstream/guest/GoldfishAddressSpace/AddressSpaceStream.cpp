/*
 * Copyright 2020 Google
 * SPDX-License-Identifier: MIT
 */
#include "AddressSpaceStream.h"
#include <time.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "VirtGpu.h"
#include "util/log.h"
#include "util/perf/cpu_trace.h"

static const size_t kReadSize = 512 * 1024;
static const size_t kWriteOffset = kReadSize;

AddressSpaceStream::AddressSpaceStream(address_space_handle_t handle, uint32_t version,
                                       struct asg_context context, uint64_t ringOffset,
                                       uint64_t writeBufferOffset, struct address_space_ops ops)
    : IOStream(context.ring_config->flush_interval),
      m_ops(ops),
      m_tmpBuf(0),
      m_tmpBufSize(0),
      m_tmpBufXferSize(0),
      m_usingTmpBuf(0),
      m_readBuf(0),
      m_read(0),
      m_readLeft(0),
      m_handle(handle),
      m_version(version),
      m_context(context),
      m_ringOffset(ringOffset),
      m_writeBufferOffset(writeBufferOffset),
      m_writeBufferSize(context.ring_config->buffer_size),
      m_writeBufferMask(m_writeBufferSize - 1),
      m_buf((unsigned char*)context.buffer),
      m_writeStart(m_buf),
      m_writeStep(context.ring_config->flush_interval),
      m_notifs(0),
      m_written(0),
      m_backoffIters(0),
      m_backoffFactor(1),
      m_ringStorageSize(sizeof(struct asg_ring_storage) + m_writeBufferSize) {
    // We'll use this in the future, but at the moment,
    // it's a potential compile Werror.
    (void)m_ringStorageSize;
    (void)m_version;
}

AddressSpaceStream::~AddressSpaceStream() {
    flush();
    ensureType3Finished();
    ensureType1Finished();

    if (!m_mapping) {
        m_ops.unmap(m_context.to_host, sizeof(struct asg_ring_storage));
        m_ops.unmap(m_context.buffer, m_writeBufferSize);
        m_ops.unclaim_shared(m_handle, m_ringOffset);
        m_ops.unclaim_shared(m_handle, m_writeBufferOffset);
    }

    m_ops.close(m_handle);
    if (m_readBuf) free(m_readBuf);
    if (m_tmpBuf) free(m_tmpBuf);
}

size_t AddressSpaceStream::idealAllocSize(size_t len) {
    if (len > m_writeStep) return len;
    return m_writeStep;
}

void* AddressSpaceStream::allocBuffer(size_t minSize) {
    MESA_TRACE_SCOPE("allocBuffer");
    ensureType3Finished();

    if (!m_readBuf) {
        m_readBuf = (unsigned char*)malloc(kReadSize);
    }

    size_t allocSize =
        (m_writeStep < minSize ? minSize : m_writeStep);

    if (m_writeStep < allocSize) {
        if (!m_tmpBuf) {
            m_tmpBufSize = allocSize * 2;
            m_tmpBuf = (unsigned char*)malloc(m_tmpBufSize);
        }

        if (m_tmpBufSize < allocSize) {
            m_tmpBufSize = allocSize * 2;
            m_tmpBuf = (unsigned char*)realloc(m_tmpBuf, m_tmpBufSize);
        }

        if (!m_usingTmpBuf) {
            flush();
        }

        m_usingTmpBuf = true;
        m_tmpBufXferSize = allocSize;
        return m_tmpBuf;
    } else {
        if (m_usingTmpBuf) {
            writeFully(m_tmpBuf, m_tmpBufXferSize);
            m_usingTmpBuf = false;
            m_tmpBufXferSize = 0;
        }

        return m_writeStart;
    }
}

int AddressSpaceStream::commitBuffer(size_t size)
{
    if (size == 0) return 0;

    if (m_usingTmpBuf) {
        writeFully(m_tmpBuf, size);
        m_tmpBufXferSize = 0;
        m_usingTmpBuf = false;
        return 0;
    } else {
        int res = type1Write(m_writeStart - m_buf, size);
        advanceWrite();
        return res;
    }
}

namespace {

// Where the guest's time goes while it is not making progress.
//
// The guest render thread spends roughly half its time in the transport, and "in the transport"
// covers three different waits that need different fixes, so lumping them together says nothing:
//
//   reply   -- blocked in read() for a host answer to a call that carries a return value. Costs a
//              full round trip, and the fix is to stop needing the answer.
//   type1   -- draining the small-command ring before the global transfer_mode can be flipped.
//              Cost is proportional to what is queued, and the fix is a per-transfer mode.
//   type3   -- draining a large transfer the same way.
//
// Enabled by GFXSTREAM_STREAM_PROFILE=1, reported every GFXSTREAM_STREAM_PROFILE_SEC seconds
// (default 10) per thread, since each guest thread has its own ring and they do not share a
// bottleneck.
struct StreamWaitProfile {
    struct Bucket {
        uint64_t nanos = 0;
        uint64_t count = 0;
        uint64_t maxNanos = 0;
        // Where the wait time sits, not just its mean. A reply wait made of queueing behind
        // other work spreads roughly evenly from zero up to however long that work takes; one
        // made of a fixed cost (a wakeup, a doorbell round trip) clusters at that cost. The mean
        // is the same in both cases and the fix is not: the first wants the call to stop waiting,
        // the second wants the wakeup to be cheaper.
        static constexpr uint64_t kEdgesUs[6] = {25, 50, 100, 200, 400, 800};
        uint64_t hist[7] = {};
        void add(uint64_t dt) {
            nanos += dt;
            ++count;
            if (dt > maxNanos) maxNanos = dt;
            const uint64_t us = dt / 1000;
            size_t i = 0;
            while (i < 6 && us >= kEdgesUs[i]) ++i;
            ++hist[i];
        }
    };
    Bucket reply, type1, type3;
    uint64_t lastReport = 0;
    double reportSec = 10.0;
    bool enabled = false;
    bool init = false;
};

thread_local StreamWaitProfile tStreamProf;

uint64_t streamProfNow() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

// 0 when disabled, so the caller pays one compare and no clock read.
uint64_t streamProfBegin() {
    StreamWaitProfile& p = tStreamProf;
    if (!p.init) {
        p.init = true;
        const char* env = getenv("GFXSTREAM_STREAM_PROFILE");
        p.enabled = env && env[0] != '0';
        if (const char* sec = getenv("GFXSTREAM_STREAM_PROFILE_SEC")) {
            const double v = atof(sec);
            if (v > 0.0) p.reportSec = v;
        }
        if (p.enabled) p.lastReport = streamProfNow();
    }
    return p.enabled ? streamProfNow() : 0;
}

void streamProfReport(StreamWaitProfile& p, uint64_t now) {
    const double elapsed = (now - p.lastReport) / 1e9;
    if (elapsed < p.reportSec) return;
    p.lastReport = now;
    const uint64_t total = p.reply.nanos + p.type1.nanos + p.type3.nanos;
    if (!total) return;
    char replyHist[160];
    snprintf(replyHist, sizeof(replyHist),
             " reply-us <25:%llu <50:%llu <100:%llu <200:%llu <400:%llu <800:%llu 800+:%llu",
             (unsigned long long)p.reply.hist[0], (unsigned long long)p.reply.hist[1],
             (unsigned long long)p.reply.hist[2], (unsigned long long)p.reply.hist[3],
             (unsigned long long)p.reply.hist[4], (unsigned long long)p.reply.hist[5],
             (unsigned long long)p.reply.hist[6]);
    mesa_logi(
        "STREAMPROF tid-local over %.1fs: waiting %.1fms/s (%.1f%% of the thread) -- "
        "reply %.1fms/s (n=%.0f/s avg=%lluus max=%lluus) "
        "type1-drain %.1fms/s (n=%.0f/s avg=%lluus max=%lluus) "
        "type3-drain %.1fms/s (n=%.0f/s avg=%lluus max=%lluus)",
        elapsed, total / elapsed / 1e6, 100.0 * total / elapsed / 1e9,
        p.reply.nanos / elapsed / 1e6, p.reply.count / elapsed,
        (unsigned long long)(p.reply.count ? p.reply.nanos / p.reply.count / 1000 : 0),
        (unsigned long long)(p.reply.maxNanos / 1000),
        p.type1.nanos / elapsed / 1e6, p.type1.count / elapsed,
        (unsigned long long)(p.type1.count ? p.type1.nanos / p.type1.count / 1000 : 0),
        (unsigned long long)(p.type1.maxNanos / 1000),
        p.type3.nanos / elapsed / 1e6, p.type3.count / elapsed,
        (unsigned long long)(p.type3.count ? p.type3.nanos / p.type3.count / 1000 : 0),
        (unsigned long long)(p.type3.maxNanos / 1000));
    mesa_logi("STREAMPROF%s", replyHist);
    p.reply = {};
    p.type1 = {};
    p.type3 = {};
}

void streamProfEnd(StreamWaitProfile::Bucket StreamWaitProfile::*which, uint64_t start) {
    if (!start) return;
    StreamWaitProfile& p = tStreamProf;
    const uint64_t now = streamProfNow();
    (p.*which).add(now - start);
    streamProfReport(p, now);
}

}  // namespace

const unsigned char *AddressSpaceStream::readFully(void *ptr, size_t totalReadSize)
{

    unsigned char* userReadBuf = static_cast<unsigned char*>(ptr);

    if (!userReadBuf) {
        if (totalReadSize > 0) {
            mesa_loge(
                "AddressSpaceStream::commitBufferAndReadFully failed, userReadBuf=NULL, "
                "totalReadSize %zu, lethal"
                " error, exiting.",
                totalReadSize);
            abort();
        }
        return nullptr;
    }

    // Advance buffered read if not yet consumed.
    size_t remaining = totalReadSize;
    size_t bufferedReadSize =
        m_readLeft < remaining ? m_readLeft : remaining;

    if (bufferedReadSize) {
        memcpy(userReadBuf,
               m_readBuf + (m_read - m_readLeft),
               bufferedReadSize);
        remaining -= bufferedReadSize;
        m_readLeft -= bufferedReadSize;
    }

    if (!remaining) return userReadBuf;

    // Read up to kReadSize bytes if all buffered read has been consumed. Everything past this
    // point is the guest blocked on the host: time it as the "reply" bucket.
    size_t maxRead = m_readLeft ? 0 : kReadSize;
    ssize_t actual = 0;
    const uint64_t replyWaitStart = streamProfBegin();

    if (maxRead) {
        actual = speculativeRead(m_readBuf, maxRead);

        // Updated buffered read size.
        if (actual > 0) {
            m_read = m_readLeft = actual;
        }

        if (actual == 0) {
            mesa_logd("%s: end of pipe", __FUNCTION__);
            return NULL;
        }
    }

    // Consume buffered read and read more if necessary.
    while (remaining) {
        bufferedReadSize = m_readLeft < remaining ? m_readLeft : remaining;
        if (bufferedReadSize) {
            memcpy(userReadBuf + (totalReadSize - remaining),
                   m_readBuf + (m_read - m_readLeft),
                   bufferedReadSize);
            remaining -= bufferedReadSize;
            m_readLeft -= bufferedReadSize;
            continue;
        }

        actual = speculativeRead(m_readBuf, kReadSize);

        if (actual == 0) {
            mesa_logd("%s: Failed reading from pipe: %d", __FUNCTION__, errno);
            return NULL;
        }

        if (actual > 0) {
            m_read = m_readLeft = actual;
            continue;
        }
    }

    streamProfEnd(&StreamWaitProfile::reply, replyWaitStart);
    resetBackoff();
    return userReadBuf;
}

const unsigned char *AddressSpaceStream::read(void *buf, size_t *inout_len) {
    unsigned char* dst = (unsigned char*)buf;
    size_t wanted = *inout_len;
    ssize_t actual = speculativeRead(dst, wanted);

    if (actual >= 0) {
        *inout_len = actual;
    } else {
        return nullptr;
    }

    return (const unsigned char*)dst;
}

int AddressSpaceStream::writeFully(const void* buf, size_t size) {
    MESA_TRACE_SCOPE("writeFully");
    ensureType3Finished();
    ensureType1Finished();

    m_context.ring_config->transfer_size = size;
    m_context.ring_config->transfer_mode = 3;

    size_t sent = 0;
    size_t preferredChunkSize = m_writeBufferSize / 4;
    size_t chunkSize = size < preferredChunkSize ? size : preferredChunkSize;
    const uint8_t* bufferBytes = (const uint8_t*)buf;

    bool hostPinged = false;
    while (sent < size) {
        size_t remaining = size - sent;
        size_t sendThisTime = remaining < chunkSize ? remaining : chunkSize;

        long sentChunks =
            ring_buffer_view_write(
                m_context.to_host_large_xfer.ring,
                &m_context.to_host_large_xfer.view,
                bufferBytes + sent, sendThisTime, 1);

        if (!hostPinged && *(m_context.host_state) != ASG_HOST_STATE_CAN_CONSUME &&
            *(m_context.host_state) != ASG_HOST_STATE_RENDERING) {
            notifyAvailable();
            hostPinged = true;
        }

        if (sentChunks == 0) {
            ring_buffer_yield();
            backoff();
        }

        sent += sentChunks * sendThisTime;

        if (isInError()) {
            return -1;
        }
    }

    bool isRenderingAfter = ASG_HOST_STATE_RENDERING == __atomic_load_n(m_context.host_state, __ATOMIC_ACQUIRE);

    if (!isRenderingAfter) {
        notifyAvailable();
    }

    ensureType3Finished();

    resetBackoff();
    m_context.ring_config->transfer_mode = 1;
    m_written += size;

    float mb = (float)m_written / 1048576.0f;
    if (mb > 100.0f) {
        mesa_logd("%s: %f mb in %d notifs. %f mb/notif\n", __func__, mb, m_notifs,
                  m_notifs ? mb / (float)m_notifs : 0.0f);
        m_notifs = 0;
        m_written = 0;
    }
    return 0;
}

int AddressSpaceStream::writeFullyAsync(const void* buf, size_t size) {
    MESA_TRACE_SCOPE("writeFullyAsync");
    ensureType3Finished();
    ensureType1Finished();

    __atomic_store_n(&m_context.ring_config->transfer_size, size, __ATOMIC_RELEASE);
    m_context.ring_config->transfer_mode = 3;

    size_t sent = 0;
    size_t preferredChunkSize = m_writeBufferSize / 2;
    size_t chunkSize = size < preferredChunkSize ? size : preferredChunkSize;
    const uint8_t* bufferBytes = (const uint8_t*)buf;

    bool pingedHost = false;

    while (sent < size) {
        size_t remaining = size - sent;
        size_t sendThisTime = remaining < chunkSize ? remaining : chunkSize;

        long sentChunks =
            ring_buffer_view_write(
                m_context.to_host_large_xfer.ring,
                &m_context.to_host_large_xfer.view,
                bufferBytes + sent, sendThisTime, 1);

        uint32_t hostState = __atomic_load_n(m_context.host_state, __ATOMIC_ACQUIRE);

        if ((!pingedHost &&
             hostState != ASG_HOST_STATE_CAN_CONSUME &&
             hostState != ASG_HOST_STATE_RENDERING) ||
            deepWaitWantsPing()) {
            pingedHost = true;
            notifyAvailable();
        }

        if (sentChunks == 0) {
            ring_buffer_yield();
            backoff();
        }

        sent += sentChunks * sendThisTime;

        if (isInError()) {
            return -1;
        }
    }


    bool isRenderingAfter = ASG_HOST_STATE_RENDERING == __atomic_load_n(m_context.host_state, __ATOMIC_ACQUIRE);

    if (!isRenderingAfter) {
        notifyAvailable();
    }

    resetBackoff();
    m_context.ring_config->transfer_mode = 1;
    m_written += size;

    float mb = (float)m_written / 1048576.0f;
    if (mb > 100.0f) {
        mesa_logd("%s: %f mb in %d notifs. %f mb/notif\n", __func__, mb, m_notifs,
                  m_notifs ? mb / (float)m_notifs : 0.0f);
        m_notifs = 0;
        m_written = 0;
    }
    return 0;
}

const unsigned char *AddressSpaceStream::commitBufferAndReadFully(
    size_t writeSize, void *userReadBufPtr, size_t totalReadSize) {

    if (m_usingTmpBuf) {
        writeFully(m_tmpBuf, writeSize);
        m_usingTmpBuf = false;
        m_tmpBufXferSize = 0;
        return readFully(userReadBufPtr, totalReadSize);
    } else {
        commitBuffer(writeSize);
        return readFully(userReadBufPtr, totalReadSize);
    }
}

bool AddressSpaceStream::isInError() const {
    return 1 == m_context.ring_config->in_error;
}

ssize_t AddressSpaceStream::speculativeRead(unsigned char* readBuffer, size_t trySize) {
    ensureType3Finished();
    ensureType1Finished();

    size_t actuallyRead = 0;

    // Wake the consumer once if it has parked, the same way ensureType1Finished() does.
    //
    // Normally unnecessary: the host is mid-command when it owes a reply, so it is not parked. But
    // if it ever does park with a reply outstanding, nothing here brings it back -- this loop only
    // spins, and there is no host-to-guest doorbell. That turns a recoverable desync into a hang:
    // observed after a stream corruption, gnome-shell burned 580% sys in backoff() while every
    // host render thread slept, and only a full VM restart cleared it. A single ping costs one
    // virtio round trip on a path that is already stuck, and makes the failure survivable.
    bool pingedHost = false;

    while (!actuallyRead) {

        uint32_t readAvail =
            ring_buffer_available_read(
                m_context.from_host_large_xfer.ring,
                &m_context.from_host_large_xfer.view);

        if (!readAvail) {
            ring_buffer_yield();
            backoff();
            if ((!pingedHost && *(m_context.host_state) != ASG_HOST_STATE_CAN_CONSUME &&
                 *(m_context.host_state) != ASG_HOST_STATE_RENDERING) ||
                deepWaitWantsPing()) {
                notifyAvailable();
                pingedHost = true;
            }
            continue;
        }

        uint32_t toRead = readAvail > trySize ?  trySize : readAvail;

        long stepsRead = ring_buffer_view_read(
            m_context.from_host_large_xfer.ring,
            &m_context.from_host_large_xfer.view,
            readBuffer, toRead, 1);

        actuallyRead += stepsRead * toRead;

        if (isInError()) {
            return -1;
        }
    }

    return actuallyRead;
}

void AddressSpaceStream::notifyAvailable() {
    MESA_TRACE_SCOPE("PING");
    struct address_space_ping request;
    request.metadata = ASG_NOTIFY_AVAILABLE;
    request.resourceId = m_resourceId;
    m_ops.ping(m_handle, &request);
    ++m_notifs;
}

uint32_t AddressSpaceStream::getRelativeBufferPos(uint32_t pos) {
    return pos & m_writeBufferMask;
}

void AddressSpaceStream::advanceWrite() {
    m_writeStart += m_context.ring_config->flush_interval;

    if (m_writeStart == m_buf + m_context.ring_config->buffer_size) {
        m_writeStart = m_buf;
    }
}

void AddressSpaceStream::ensureConsumerFinishing() {
    uint32_t currAvailRead = ring_buffer_available_read(m_context.to_host, 0);

    while (currAvailRead) {
        ring_buffer_yield();
        uint32_t nextAvailRead = ring_buffer_available_read(m_context.to_host, 0);

        if (nextAvailRead != currAvailRead) {
            break;
        }

        if (*(m_context.host_state) != ASG_HOST_STATE_CAN_CONSUME &&
            *(m_context.host_state) != ASG_HOST_STATE_RENDERING) {
            notifyAvailable();
            break;
        }

        backoff();
    }
}

void AddressSpaceStream::ensureType1Finished() {
    MESA_TRACE_SCOPE("ensureType1Finished");
    const uint64_t drainStart = streamProfBegin();

    uint32_t currAvailRead =
        ring_buffer_available_read(m_context.to_host, 0);

    // Wake the consumer once if it has parked itself, the way ensureConsumerFinishing() does.
    // This wait sits at the head of every writeFullyAsync() -- the path a command buffer takes
    // to the host -- and spinning alone will never bring back a consumer that is asleep. Once
    // is the operative word: a ping is a virtio round trip, so pinging per iteration turns the
    // wait into a storm of them.
    bool pingedHost = false;
    while (currAvailRead) {
        backoff();
        ring_buffer_yield();
        currAvailRead = ring_buffer_available_read(m_context.to_host, 0);
        if ((!pingedHost && *(m_context.host_state) != ASG_HOST_STATE_CAN_CONSUME &&
             *(m_context.host_state) != ASG_HOST_STATE_RENDERING) ||
            deepWaitWantsPing()) {
            notifyAvailable();
            pingedHost = true;
        }
        if (isInError()) {
            streamProfEnd(&StreamWaitProfile::type1, drainStart);
            return;
        }
    }
    streamProfEnd(&StreamWaitProfile::type1, drainStart);
}

void AddressSpaceStream::ensureType3Finished() {
    MESA_TRACE_SCOPE("ensureType3Finished");
    const uint64_t drainStart = streamProfBegin();
    uint32_t availReadLarge =
        ring_buffer_available_read(
            m_context.to_host_large_xfer.ring,
            &m_context.to_host_large_xfer.view);
    while (availReadLarge) {
        ring_buffer_yield();
        backoff();
        availReadLarge =
            ring_buffer_available_read(
                m_context.to_host_large_xfer.ring,
                &m_context.to_host_large_xfer.view);
        if (*(m_context.host_state) != ASG_HOST_STATE_CAN_CONSUME &&
            *(m_context.host_state) != ASG_HOST_STATE_RENDERING) {
            notifyAvailable();
        }
        if (isInError()) {
            streamProfEnd(&StreamWaitProfile::type3, drainStart);
            return;
        }
    }
    streamProfEnd(&StreamWaitProfile::type3, drainStart);
}

int AddressSpaceStream::type1Write(uint32_t bufferOffset, size_t size) {
    MESA_TRACE_SCOPE("type1Write");

    ensureType3Finished();

    size_t sent = 0;
    size_t sizeForRing = sizeof(struct asg_type1_xfer);

    struct asg_type1_xfer xfer = {
        bufferOffset,
        (uint32_t)size,
    };

    uint8_t* writeBufferBytes = (uint8_t*)(&xfer);

    uint32_t maxOutstanding = 1;
    uint32_t maxSteps = m_context.ring_config->buffer_size /
            m_context.ring_config->flush_interval;

    if (maxSteps > 1) maxOutstanding = maxSteps - 1;

    uint32_t ringAvailReadNow = ring_buffer_available_read(m_context.to_host, 0);

    while (ringAvailReadNow >= maxOutstanding * sizeForRing) {
        ringAvailReadNow = ring_buffer_available_read(m_context.to_host, 0);
    }

    bool hostPinged = false;
    while (sent < sizeForRing) {

        long sentChunks = ring_buffer_write(
            m_context.to_host,
            writeBufferBytes + sent,
            sizeForRing - sent, 1);

        if (!hostPinged &&
            *(m_context.host_state) != ASG_HOST_STATE_CAN_CONSUME &&
            *(m_context.host_state) != ASG_HOST_STATE_RENDERING) {
            notifyAvailable();
            hostPinged = true;
        }

        if (sentChunks == 0) {
            ring_buffer_yield();
            backoff();
        }

        sent += sentChunks * (sizeForRing - sent);

        if (isInError()) {
            return -1;
        }
    }

    bool isRenderingAfter = ASG_HOST_STATE_RENDERING == __atomic_load_n(m_context.host_state, __ATOMIC_ACQUIRE);

    if (!isRenderingAfter) {
        notifyAvailable();
    }

    m_written += size;

    float mb = (float)m_written / 1048576.0f;
    if (mb > 100.0f) {
        mesa_logd("%s: %f mb in %d notifs. %f mb/notif\n", __func__, mb, m_notifs,
                  m_notifs ? mb / (float)m_notifs : 0.0f);
        m_notifs = 0;
        m_written = 0;
    }

    resetBackoff();
    return 0;
}

// Shared with deepWaitWantsPing(), which needs to know when backoff() has stopped spinning and
// started sleeping.
static const uint32_t kBackoffItersThreshold = 50000000;

// Ask for another ping, roughly once a second, but only once the wait is deep.
//
// The wait loops ping the host once when it looks parked. Once is right for the common case and
// wrong for the one that matters: the ping can be lost, the host can park again after it, and the
// state word can say CAN_CONSUME or RENDERING while the thread that would consume is asleep -- in
// which case the guarded one-shot never fires at all. There is no host-to-guest doorbell, so
// nothing else ends the wait, and a missed wakeup becomes a permanent hang. Observed on KDE:
// plasmashell parked in vkCreateSemaphore inside kopper_acquire while every host render thread
// slept at 0% CPU, and the desktop never drew.
//
// Rate is the whole design. backoff() spins for its first 50M turns and only then starts sleeping
// up to 1ms a turn, so gating on that keeps every ping out of the fast path; one per ~1000 sleeping
// turns is about a second. Pinging per iteration was tried and is much worse than not pinging --
// each one is a virtio round trip, and it cost Minecraft two thirds of its frame rate.
bool AddressSpaceStream::deepWaitWantsPing() {
    if (m_backoffIters <= kBackoffItersThreshold) return false;
    if (++m_deepWaitPingCountdown < 1000) return false;
    m_deepWaitPingCountdown = 0;
    return true;
}

void AddressSpaceStream::backoff() {
    static const uint32_t kBackoffFactorDoublingIncrement = 50000000;
    ++m_backoffIters;

    if (m_backoffIters > kBackoffItersThreshold) {
        usleep(m_backoffFactor);
        uint32_t itersSoFarAfterThreshold = m_backoffIters - kBackoffItersThreshold;
        if (itersSoFarAfterThreshold > kBackoffFactorDoublingIncrement) {
            m_backoffFactor = m_backoffFactor << 1;
            if (m_backoffFactor > 1000) m_backoffFactor = 1000;
            m_backoffIters = kBackoffItersThreshold;
        }
    }
}

void AddressSpaceStream::resetBackoff() {
    m_backoffIters = 0;
    m_backoffFactor = 1;
}
