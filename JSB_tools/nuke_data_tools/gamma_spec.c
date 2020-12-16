//#define gamma_spec_cxx
//#include "gamma_spec.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

TTree *g_tree;

TBranch        *b_erg;   //!
TBranch        *b_sigma_erg;   //!
TBranch        *b_t_min;   //!
TBranch        *b_t_max;   //!
TBranch        *b_shot;   //!
TBranch        *b_t;   //!
TBranch        *b_eff;   //!
TBranch        *b_eff_err;   //!
TBranch        *b_J;   //!

Double_t        erg;
Double_t        sigma_erg;
Double_t        t_min;
Double_t        t_max;
Double_t        shot;
Double_t        t;
Double_t        eff;
Double_t        eff_err;

int main(){
    cout<<"Main!"<<endl;
    return 1;
}

int set_up(TTree *tree){
    g_tree = tree;
   tree->SetBranchAddress("erg", &erg, &b_erg);
   tree->SetBranchAddress("sigma_erg", &sigma_erg, &b_sigma_erg);
   tree->SetBranchAddress("t_min", &t_min, &b_t_min);
   tree->SetBranchAddress("t_max", &t_max, &b_t_max);
   tree->SetBranchAddress("shot", &shot, &b_shot);
   tree->SetBranchAddress("t", &t, &b_t);
   tree->SetBranchAddress("eff", &eff, &b_eff);
   tree->SetBranchAddress("eff_err", &eff_err, &b_eff_err);
   return 1;
}

float get_max_time(){
    float max_time = 0;
    for (int i=0; i<g_tree->GetEntries(); i++){
        g_tree->GetEntry(i);
        if (t>max_time){
            max_time = t;
        }
    }
    return max_time;
}
