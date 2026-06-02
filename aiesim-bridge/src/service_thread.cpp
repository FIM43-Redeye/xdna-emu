#include "service_thread.h"

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

void Service::set_wake(std::function<void()> wake) {
    std::lock_guard<std::mutex> lock(mtx_);
    wake_ = std::move(wake);
}

void* Service::start(const char* arch, const char* device_json) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (started_) {
        return elab_ok_ ? top_ : nullptr;  // one sim per process
    }
    started_ = true;
    g_aiesim_arch = arch;
    g_aiesim_device_json = device_json;
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
    std::function<void()> wake;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!elab_ok_ || joined_) {
            c.reply_int = 1;  // service not up / torn down -> fail fast
            return;
        }
        queue_.push_back(&c);
        wake = wake_;
    }
    q_cv_.notify_one();      // wake sc_main's wait_for_pending so it sc_starts
    if (wake) wake();        // ring the Doorbell so the driver runs within sc_start

    std::unique_lock<std::mutex> clock(c.m);
    c.cv.wait(clock, [&c] { return c.done; });
}

Command* Service::try_pop() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.empty()) return nullptr;
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

void Service::mark_shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);
    shutdown_ = true;
}

void Service::wait_for_pending() {
    std::unique_lock<std::mutex> lock(mtx_);
    q_cv_.wait(lock, [this] { return !queue_.empty(); });
}

bool Service::is_shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);
    return shutdown_;
}

void Service::shutdown_and_join() {
    std::function<void()> wake;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!started_ || !elab_ok_ || joined_) return;
        joined_ = true;  // claim the shutdown
        wake = wake_;
    }

    // Enqueue SHUTDOWN directly (bypass submit()'s joined_ guard, which we just
    // set) so the driver unwinds sc_main cleanly (sc_stop -> end_of_simulation).
    Command stop(Command::SHUTDOWN);
    {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push_back(&stop);
    }
    q_cv_.notify_one();
    if (wake) wake();
    {
        std::unique_lock<std::mutex> clock(stop.m);
        stop.cv.wait(clock, [&stop] { return stop.done; });
    }

    if (thread_.joinable()) thread_.join();
}

}  // namespace aiesim
