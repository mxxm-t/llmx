#include "backends/cpu/cpu_backend.hpp"
#include <iostream>
int main() {
    backend::RowRun runs[2]={{3,1},{2,2}};
    size_t visited=0;
    bool threw=false;
    try { backend::CpuBackend::each_run(2,{runs,2},[&](size_t,size_t n,bool) { visited+=n; }); }
    catch (const std::runtime_error&) { threw=true; }
    std::cout << "mixed_invalid_runs batch=2 callback_rows=" << visited << " threw_after_callback=" << threw << '\n';
    runs[1].extent=1;
    size_t homogeneous=0;
    bool rejected=false;
    try { backend::CpuBackend::each_run(2,{runs,2},[&](size_t,size_t n,bool) { homogeneous+=n; }); }
    catch (const std::runtime_error&) { rejected=true; }
    std::cout << "same_kind_invalid_runs batch=2 callback_rows=" << homogeneous << " rejected=" << rejected << '\n';
    return visited==3 && threw && homogeneous==2 && !rejected ? 0 : 1;
}
