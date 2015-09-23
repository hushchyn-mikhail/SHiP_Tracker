#include <stdlib.h>

void run() {
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000000000);

  TString tree_location="cbmsim";
  TChain * chain = new TChain(tree_location);
  chain->AddFile("ship.10.0.Pythia8-TGeant4.root");
  gROOT->ProcessLine(".L selection.C+");
  cout<<"entries in chain = "<<chain->GetEntries()<<endl;
  selection * read = new selection(chain);
  read->Loop();

}
