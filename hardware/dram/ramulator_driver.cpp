#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace {

struct TraceRequest {
  int type;
  Ramulator::Addr_t address;
};

std::vector<TraceRequest> read_trace(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open trace: " + path);
  std::vector<TraceRequest> requests;
  std::string operation;
  std::string address;
  while (stream >> operation >> address) {
    int type;
    if (operation == "LD") type = Ramulator::Request::Type::Read;
    else if (operation == "ST") type = Ramulator::Request::Type::Write;
    else throw std::runtime_error("unsupported trace operation: " + operation);
    requests.push_back({type, static_cast<Ramulator::Addr_t>(std::stoull(address, nullptr, 0))});
  }
  if (requests.empty()) throw std::runtime_error("trace is empty");
  return requests;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: ramulator_driver CONFIG TRACE METRICS STATS\n";
    return 2;
  }
  try {
    auto config = Ramulator::Config::parse_config_file(argv[1]);
    std::unique_ptr<Ramulator::IFrontEnd> frontend(Ramulator::Factory::create_frontend(config));
    std::unique_ptr<Ramulator::IMemorySystem> memory(Ramulator::Factory::create_memory_system(config));
    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());

    const auto requests = read_trace(argv[2]);
    const int transaction_bytes = memory->get_tx_bytes();
    std::size_t submitted = 0;
    std::size_t completed = 0;
    std::size_t reads = 0;
    std::size_t writes = 0;
    std::uint64_t read_latency = 0;
    std::uint64_t cycles = 0;
    const std::uint64_t max_cycles = std::max<std::uint64_t>(1000000, requests.size() * 100000);

    while (completed < requests.size()) {
      if (submitted < requests.size()) {
        const auto request = requests[submitted];
        auto callback = [&](Ramulator::Request& done) {
          completed++;
          if (done.type_id == Ramulator::Request::Type::Read) {
            reads++;
            if (done.arrive >= 0 && done.depart >= done.arrive) {
              read_latency += static_cast<std::uint64_t>(done.depart - done.arrive);
            }
          } else {
            writes++;
          }
        };
        if (frontend->receive_external_requests(
                request.type, request.address, 0, callback, transaction_bytes)) {
          submitted++;
        }
      }
      memory->tick();
      cycles++;
      if (cycles > max_cycles) throw std::runtime_error("request drain exceeded safety bound");
    }

    frontend->finalize();
    memory->finalize();
    std::ofstream stats(argv[4]);
    memory->print_stats(stats);
    std::ofstream metrics(argv[3]);
    metrics << std::setprecision(12)
            << "{\n"
            << "  \"memory_cycles\": " << cycles << ",\n"
            << "  \"submitted_requests\": " << submitted << ",\n"
            << "  \"completed_requests\": " << completed << ",\n"
            << "  \"read_requests\": " << reads << ",\n"
            << "  \"write_requests\": " << writes << ",\n"
            << "  \"average_read_latency_cycles\": "
            << (reads ? static_cast<double>(read_latency) / reads : 0.0) << ",\n"
            << "  \"transaction_bytes\": " << transaction_bytes << "\n"
            << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 2;
  }
}
