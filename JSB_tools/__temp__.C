#define __temp___cxx
#include "__temp__.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void __temp__::Loop(TH1F *h0, TH1F *h1, int max_entries)
{
//   In a ROOT session, you can do:
//      root> .L __temp__.C
//      root> __temp__ t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;


	  if (((217.24996079102698 <= (erg) && (erg) <= 221.26459693294146)) && (((0 <= (t_rnd) && (t_rnd) <= 400)))){h0->Fill(t_rnd, (1/eff));}

	  if ((((221.26459693294146 <= (erg) && (erg) <= 223.27191500389873)) || ((215.2426427200697 <= (erg) && (erg) <= 217.24996079102698))) && (((0 <= (t_rnd) && (t_rnd) <= 400)))){h1->Fill(t_rnd, (1/eff));}
	  if (jentry > max_entries){break;}
   }
}
