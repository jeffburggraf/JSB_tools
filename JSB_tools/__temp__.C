#define __temp___cxx
#include "__temp__.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void __temp__::Loop(TH1F *h0, TH1F *h1, TH1F *h2, TH1F *h3, TH1F *h4, TH1F *h5, TH1F *h6, TH1F *h7, TH1F *h8, TH1F *h9, TH1F *h10, int max_entries)
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


	  if (shot == 42 && (10 <= (t_rnd) && (t_rnd) <= 200)){h0->Fill(erg);}

	  if (shot == 43 && (10 <= (t_rnd) && (t_rnd) <= 200)){h1->Fill(erg);}

	  if (shot == 44 && (10 <= (t_rnd) && (t_rnd) <= 200)){h2->Fill(erg);}

	  if (shot == 45 && (10 <= (t_rnd) && (t_rnd) <= 200)){h3->Fill(erg);}

	  if (shot == 46 && (10 <= (t_rnd) && (t_rnd) <= 200)){h4->Fill(erg);}

	  if (shot == 47 && (10 <= (t_rnd) && (t_rnd) <= 200)){h5->Fill(erg);}

	  if (shot == 48 && (10 <= (t_rnd) && (t_rnd) <= 200)){h6->Fill(erg);}

	  if (shot == 49 && (10 <= (t_rnd) && (t_rnd) <= 200)){h7->Fill(erg);}

	  if (shot == 50 && (10 <= (t_rnd) && (t_rnd) <= 200)){h8->Fill(erg);}

	  if (shot == 51 && (10 <= (t_rnd) && (t_rnd) <= 200)){h9->Fill(erg);}

	  if (shot == 52 && (10 <= (t_rnd) && (t_rnd) <= 200)){h10->Fill(erg);}
	  if (jentry > max_entries){break;}
   }
}
