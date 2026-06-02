#include "service_thread.h"

// The bootstrap globals + kernel entry the service thread drives. Declared here
// (not via a header) to keep this translation unit free of SystemC includes;
// they are defined in sc_bootstrap.cpp.
extern "C" {
extern const char* g_aiesim_arch;
extern const char* g_aiesim_device_json;
int aiesim_bridge_start_systemc();  // runs sc_elab_and_sim -> sc_main
}

namespace aiesim {

Service& Service::instance() {
    static Service s;
    return s;
}

void* Service::start(const char* arch, const char* device_json) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (started_) {
        // One SystemC sim per process: a second create returns the same cluster.
        return elab_ok_ ? top_ : nullptr;
    }
    started_ = true;

    // sc_main reads these to construct the cluster. They are caller-owned and
    // must outlive elaboration; aiesim_create holds the originals for the run.
    g_aiesim_arch = arch;
    g_aiesim_device_json = device_json;

    // Spawn the kernel thread. It runs until a SHUTDOWN command lets sc_main
    // return. publish() (called from sc_main) releases the wait below.
    thread_ = std::thread([] { aiesim_bridge_start_systemc(); });

    elab_cv_.wait(lock, [this] { return elaborated_; });
    return elab_ok_ ? top_ : nullptr;
}

void Service::publish(void* top, bool ok) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        top_ = top;
        elab_ok_ = ok;
        elaborated_ = true;
    }
    elab_cv_.notify_all();
}

void Service::submit(Command& c) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!elab_ok_ || joined_) {
            // Service never came up, or was already torn down: fail loudly
            // rather than block forever on a reply that will never arrive.
            c.reply_int = 1;
            return;
        }
        queue_.push_back(&c);
    }
    q_cv_.notify_one();

    std::unique_lock<std::mutex> clock(c.m);
    c.cv.wait(clock, [&c] { return c.done; });
}

Command* Service::next() {
    std::unique_lock<std::mutex> lock(mtx_);
    q_cv_.wait(lock, [this] { return !queue_.empty(); });
    Command* c = queue_.front();
    queue_.pop_front();
    return c;
}

void Service::complete(Command* c) {
    {
        std::lock_guard<std::mutex> lock(c->m);
        c->done = true;
    }
    c->cv.notify_one();
}

void Service::shutdown_and_join() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!started_ || !elab_ok_ || joined_) return;
        joined_ = true;  // claim the shutdown; blocks double / concurrent calls
    }

    // Enqueue SHUTDOWN directly -- we bypass submit() because we just set
    // joined_, which submit() now rejects. SHUTDOWN flows as a normal command so
    // the service thread unwinds sc_main cleanly (end_of_simulation runs).
    Command stop(Command::SHUTDOWN);
    {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push_back(&stop);
    }
    q_cv_.notify_one();
    {
        std::unique_lock<std::mutex> clock(stop.m);
        stop.cv.wait(clock, [&stop] { return stop.done; });
    }

    if (thread_.joinable()) {
        thread_.join();
    }
    // aiesim is process-singleton and not restartable in-process: after this no
    // further submits will be serviced (they fail-fast on the joined_ guard).
}

}  // namespace aiesim
