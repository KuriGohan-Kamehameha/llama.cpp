#pragma once
// Request leases live through streaming completion and websocket closure.
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

class server_drain : public std::enable_shared_from_this<server_drain> {
public:
    struct state { bool configured; bool draining; size_t active; };
    class lease {
    public:
        explicit lease(std::shared_ptr<server_drain> owner) : owner_(std::move(owner)) {}
        ~lease() { owner_->release(); }
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;
    private:
        std::shared_ptr<server_drain> owner_;
    };
    server_drain() {
        const char * value = std::getenv("LLAMA_DRAIN_MARKER");
        if (value != nullptr) marker_ = value;
    }
    std::shared_ptr<lease> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (blocked()) return {};
        ++active_;
        try { return std::make_shared<lease>(shared_from_this()); }
        catch (...) { --active_; throw; }
    }
    state snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        return {!marker_.empty(), blocked(), active_};
    }
private:
    bool blocked() const {
        if (marker_.empty()) return false;
        std::error_code ec;
        const std::filesystem::path path(marker_);
        if (!path.is_absolute() || !std::filesystem::is_directory(path.parent_path(), ec) || ec) return true;
        auto entry = std::filesystem::symlink_status(path, ec);
        if (ec == std::errc::no_such_file_or_directory) return false;
        return ec || entry.type() != std::filesystem::file_type::not_found;
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ == 0) std::terminate();
        --active_;
    }
    std::mutex mutex_;
    std::string marker_;
    size_t active_ = 0;
};
