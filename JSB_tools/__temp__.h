//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sun Nov 22 20:45:55 2020 by ROOT version 6.20/04
// from TChain tree/
//////////////////////////////////////////////////////////

#ifndef __temp___h
#define __temp___h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class __temp__ {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Double_t        erg;
   Double_t        t_min;
   Double_t        t_max;
   Double_t        shot;
   Double_t        t_rnd;
   Double_t        eff;
   Double_t        J;

   // List of branches
   TBranch        *b_erg;   //!
   TBranch        *b_t_min;   //!
   TBranch        *b_t_max;   //!
   TBranch        *b_shot;   //!
   TBranch        *b_t_rnd;   //!
   TBranch        *b_eff;   //!
   TBranch        *b_J;   //!

   __temp__(TTree *tree=0);
   virtual ~__temp__();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(TH1F *h0, TH1F *h1, int max_entries);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef __temp___cxx
__temp__::__temp__(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {

#ifdef SINGLE_TREE
      // The following code should be used if you want this class to access
      // a single tree instead of a chain
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot28.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot28.root");
      }
      f->GetObject("tree",tree);

#else // SINGLE_TREE

      // The following code should be used if you want this class to access a chain
      // of trees.
      TChain * chain = new TChain("tree","");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot5.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot6.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot8.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot9.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot10.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot11.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot12.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot13.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot14.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot18.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot19.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot20.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot21.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot22.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot23.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot24.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot27.root/tree");
      chain->Add("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Shot_data_analysis/ROOTTTrees2/_shot28.root/tree");
      tree = chain;
#endif // SINGLE_TREE

   }
   Init(tree);
}

__temp__::~__temp__()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t __temp__::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t __temp__::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void __temp__::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("erg", &erg, &b_erg);
   fChain->SetBranchAddress("t_min", &t_min, &b_t_min);
   fChain->SetBranchAddress("t_max", &t_max, &b_t_max);
   fChain->SetBranchAddress("shot", &shot, &b_shot);
   fChain->SetBranchAddress("t_rnd", &t_rnd, &b_t_rnd);
   fChain->SetBranchAddress("eff", &eff, &b_eff);
   fChain->SetBranchAddress("J", &J, &b_J);
   Notify();
}

Bool_t __temp__::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void __temp__::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t __temp__::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef __temp___cxx
