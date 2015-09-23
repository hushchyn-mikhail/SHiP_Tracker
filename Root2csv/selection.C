#define selection_cxx
#include "selection.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <fstream>

void selection::Loop()
{
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   TString outFile_name1 = "data/";
   outFile_name1 = outFile_name1 + "strawtubes.csv";
   cout<<"Start csv file"<<endl;
   cout<<"found "<<kMaxcbmroot_strawtubes_strawtubesPoint<<" points"<<endl;
   ofstream outFile1; outFile1.open(outFile_name1);



   outFile1<<"entry\tk\tUniqueID\tBits\tTrackID\tEventId\tPx\tPy\tPz\tTime\tLength\tELoss\tDetectorID\tX\tY\tZ\tPdgCode\tdist2Wire"<<endl;

   Long64_t nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     Long64_t ientry = LoadTree(jentry);
     if (ientry < 0) break;
     nb = fChain->GetEntry(jentry);

        for(int k = 0; k<kMaxcbmroot_strawtubes_strawtubesPoint; k++){
                if (strawtubesPoint_fDetectorID[k]!=0)
			outFile1<<jentry<<"\t"<<k<<"\t"<<strawtubesPoint_fUniqueID[k]<<"\t"<<strawtubesPoint_fBits[k]<<"\t"<<strawtubesPoint_fTrackID[k]<<"\t"<<strawtubesPoint_fEventId[k]<<"\t"<<strawtubesPoint_fPx[k]<<"\t"<<strawtubesPoint_fPy[k]<<"\t"<<strawtubesPoint_fPz[k]<<"\t"<<strawtubesPoint_fTime[k]<<"\t"<<strawtubesPoint_fLength[k]<<"\t"<<strawtubesPoint_fELoss[k]<<"\t"<<strawtubesPoint_fDetectorID[k]<<"\t"<<strawtubesPoint_fX[k]<<"\t"<<strawtubesPoint_fY[k]<<"\t"<<strawtubesPoint_fZ[k]<<"\t"<<strawtubesPoint_fPdgCode[k]<<"\t"<<strawtubesPoint_fdist2Wire[k]<<endl;
        }




   }
}

