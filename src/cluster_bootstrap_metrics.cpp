// Exact multinomial user-cluster bootstrap. No score binning or model inference.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
struct Row { uint32_t user; uint8_t label, end; uint16_t padding; };
static_assert(sizeof(Row)==8);
template<class T> void read(std::ifstream& in,std::vector<T>& v,size_t n){
    v.resize(n);in.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!in)throw std::runtime_error("Truncated input");
}
struct Model {std::vector<Row> ranks;std::vector<double> loss,gauc;};
void evaluate(const Model& model,const std::vector<uint32_t>& w,double total,double eligible,double* out){
    double cp=0,cn=0,gp=0,gn=0,area=0,ap=0;
    for(const auto& r:model.ranks){
        double m=w[r.user];if(r.label)gp+=m;else gn+=m;
        if(r.end){
            area+=gn*(cp+0.5*gp);cp+=gp;cn+=gn;
            if(gp>0)ap+=gp*cp/(cp+cn);
            gp=gn=0;
        }
    }
    double ll=0,ga=0;
    for(size_t u=0;u<w.size();++u){ll+=w[u]*model.loss[u];ga+=w[u]*model.gauc[u];}
    out[0]=cp>0&&cn>0?area/(cp*cn):NAN;
    out[1]=ll/total;out[2]=cp>0?ap/cp:NAN;out[3]=eligible>0?ga/eligible:NAN;
}
int main(int argc,char** argv){try{
    if(argc<6)throw std::runtime_error("usage: binary input_dir output repeats seed threads [weights.bin]");
    std::string dir=argv[1],output=argv[2];int B=std::stoi(argv[3]);uint64_t seed=std::stoull(argv[4]);int threads=std::stoi(argv[5]);
    if(B<0||threads<1||std::filesystem::exists(output))throw std::runtime_error("Invalid limits or existing output");
    std::ifstream in(dir+"/common.bin",std::ios::binary);std::vector<uint64_t> header;read(in,header,3);
    size_t N=header[0],U=header[1],M=header[2];
    std::vector<uint32_t> counts,eligible;read(in,counts,U);read(in,eligible,U);
    std::vector<Model> models(M);
    for(size_t m=0;m<M;++m){std::ifstream f(dir+"/model_"+std::to_string(m)+".bin",std::ios::binary);
        read(f,models[m].ranks,N);read(f,models[m].loss,U);read(f,models[m].gauc,U);
        for(auto& r:models[m].ranks)if(r.user>=U||r.label>1||r.end>1)throw std::runtime_error("Invalid rank row");
    }
    std::vector<double> result((B+1)*M*4);std::atomic<int> next(0),completed(0);
    auto worker=[&](){std::vector<uint32_t>w(U);for(;;){int b=next++;if(b>B)break;
        if(b==0){std::fill(w.begin(),w.end(),1);if(argc>6){std::ifstream f(argv[6],std::ios::binary);read(f,w,U);}}
        else {std::fill(w.begin(),w.end(),0);std::mt19937_64 rng(seed+uint64_t(b-1));std::uniform_int_distribution<uint32_t> draw(0,uint32_t(U-1));for(size_t u=0;u<U;++u)++w[draw(rng)];}
        double total=0,elig=0;for(size_t u=0;u<U;++u){total+=double(w[u])*counts[u];elig+=double(w[u])*eligible[u];}
        for(size_t m=0;m<M;++m)evaluate(models[m],w,total,elig,&result[(b*M+m)*4]);
        int done=++completed;if(done%100==0)std::cerr<<"completed "<<done<<"/"<<B+1<<std::endl;
    }};
    std::vector<std::thread> pool;for(int t=0;t<threads;++t)pool.emplace_back(worker);for(auto& t:pool)t.join();
    std::ofstream out(output,std::ios::binary);out.write(reinterpret_cast<char*>(result.data()),result.size()*sizeof(double));
    if(!out)throw std::runtime_error("Output write failure");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
