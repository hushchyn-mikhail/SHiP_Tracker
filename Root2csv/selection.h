//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Sep 22 17:43:12 2015 by ROOT version 5.34/26
// from TTree cbmsim//cbmroot
// found on file: ship.10.0.Pythia8-TGeant4.root
//////////////////////////////////////////////////////////

#ifndef selection_h
#define selection_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TRef.h>

// Header file for the classes stored in the TTree if any.
#include <TClonesArray.h>
#include <TObject.h>
#include <TNamed.h>

// Fixed size dimensions of array or collections stored in the TTree if any.
   const Int_t kMaxcbmroot_Stack_MCTrack = 981;
   const Int_t kMaxcbmroot_veto_vetoPoint = 12521;
   const Int_t kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint = 1991;
   const Int_t kMaxcbmroot_Target_TargetPoint = 74458;
   const Int_t kMaxcbmroot_strawtubes_strawtubesPoint = 4009;
   const Int_t kMaxcbmroot_Ecal_EcalPoint = 440;
   const Int_t kMaxcbmroot_EcalLite_EcalPointLite = 1480;
   const Int_t kMaxcbmroot_Hcal_HcalPoint = 1994;
   const Int_t kMaxcbmroot_HcalLite_HcalPointLite = 9988;
   const Int_t kMaxcbmroot_muon_muonPoint = 122;
   const Int_t kMaxcbmroot_Event_MCEventHeader = 1;

class selection {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Int_t           MCTrack_;
   UInt_t          MCTrack_fUniqueID[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   UInt_t          MCTrack_fBits[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Int_t           MCTrack_fPdgCode[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Int_t           MCTrack_fMotherId[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Double32_t      MCTrack_fPx[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Double32_t      MCTrack_fPy[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Double32_t      MCTrack_fPz[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Double32_t      MCTrack_fStartX[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Double32_t      MCTrack_fStartY[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Double32_t      MCTrack_fStartZ[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Double32_t      MCTrack_fStartT[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Double32_t      MCTrack_fW[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Int_t           MCTrack_fNPoints[kMaxcbmroot_Stack_MCTrack];   //[cbmroot.Stack.MCTrack_]
   Int_t           vetoPoint_;
   UInt_t          vetoPoint_fUniqueID[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   UInt_t          vetoPoint_fBits[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   TRef            vetoPoint_fRefToLinks[kMaxcbmroot_veto_vetoPoint];
   Int_t           vetoPoint_fTrackID[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   UInt_t          vetoPoint_fEventId[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fPx[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fPy[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fPz[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fTime[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fLength[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fELoss[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Int_t           vetoPoint_fDetectorID[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fX[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fY[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Double32_t      vetoPoint_fZ[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Int_t           vetoPoint_fPdgCode[kMaxcbmroot_veto_vetoPoint];   //[cbmroot.veto.vetoPoint_]
   Int_t           ShipRpcPoint_;
   UInt_t          ShipRpcPoint_fUniqueID[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   UInt_t          ShipRpcPoint_fBits[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   TRef            ShipRpcPoint_fRefToLinks[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];
   Int_t           ShipRpcPoint_fTrackID[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   UInt_t          ShipRpcPoint_fEventId[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fPx[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fPy[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fPz[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fTime[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fLength[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fELoss[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Int_t           ShipRpcPoint_fDetectorID[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fX[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fY[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Double32_t      ShipRpcPoint_fZ[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Int_t           ShipRpcPoint_fPdgCode[kMaxcbmroot_MagneticSpectrometer_ShipRpcPoint];   //[cbmroot.MagneticSpectrometer.ShipRpcPoint_]
   Int_t           TargetPoint_;
   UInt_t          TargetPoint_fUniqueID[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   UInt_t          TargetPoint_fBits[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   TRef            TargetPoint_fRefToLinks[kMaxcbmroot_Target_TargetPoint];
   Int_t           TargetPoint_fTrackID[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   UInt_t          TargetPoint_fEventId[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fPx[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fPy[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fPz[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fTime[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fLength[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fELoss[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Int_t           TargetPoint_fDetectorID[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fX[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fY[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Double32_t      TargetPoint_fZ[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Int_t           TargetPoint_fPdgCode[kMaxcbmroot_Target_TargetPoint];   //[cbmroot.Target.TargetPoint_]
   Int_t           strawtubesPoint_;
   UInt_t          strawtubesPoint_fUniqueID[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   UInt_t          strawtubesPoint_fBits[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   TRef            strawtubesPoint_fRefToLinks[kMaxcbmroot_strawtubes_strawtubesPoint];
   Int_t           strawtubesPoint_fTrackID[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   UInt_t          strawtubesPoint_fEventId[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fPx[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fPy[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fPz[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fTime[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fLength[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fELoss[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Int_t           strawtubesPoint_fDetectorID[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fX[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fY[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double32_t      strawtubesPoint_fZ[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Int_t           strawtubesPoint_fPdgCode[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Double_t        strawtubesPoint_fdist2Wire[kMaxcbmroot_strawtubes_strawtubesPoint];   //[cbmroot.strawtubes.strawtubesPoint_]
   Int_t           EcalPoint_;
   UInt_t          EcalPoint_fUniqueID[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   UInt_t          EcalPoint_fBits[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   TRef            EcalPoint_fRefToLinks[kMaxcbmroot_Ecal_EcalPoint];
   Int_t           EcalPoint_fTrackID[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   UInt_t          EcalPoint_fEventId[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fPx[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fPy[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fPz[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fTime[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fLength[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fELoss[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Int_t           EcalPoint_fDetectorID[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fX[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fY[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Double32_t      EcalPoint_fZ[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Int_t           EcalPoint_fPdgCode[kMaxcbmroot_Ecal_EcalPoint];   //[cbmroot.Ecal.EcalPoint_]
   Int_t           EcalPointLite_;
   UInt_t          EcalPointLite_fUniqueID[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   UInt_t          EcalPointLite_fBits[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   TRef            EcalPointLite_fRefToLinks[kMaxcbmroot_EcalLite_EcalPointLite];
   Int_t           EcalPointLite_fTrackID[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   UInt_t          EcalPointLite_fEventId[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fPx[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fPy[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fPz[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fTime[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fLength[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fELoss[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Int_t           EcalPointLite_fDetectorID[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fX[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fY[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Double32_t      EcalPointLite_fZ[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Int_t           EcalPointLite_fPdgCode[kMaxcbmroot_EcalLite_EcalPointLite];   //[cbmroot.EcalLite.EcalPointLite_]
   Int_t           HcalPoint_;
   UInt_t          HcalPoint_fUniqueID[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   UInt_t          HcalPoint_fBits[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   TRef            HcalPoint_fRefToLinks[kMaxcbmroot_Hcal_HcalPoint];
   Int_t           HcalPoint_fTrackID[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   UInt_t          HcalPoint_fEventId[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fPx[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fPy[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fPz[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fTime[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fLength[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fELoss[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Int_t           HcalPoint_fDetectorID[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fX[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fY[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Double32_t      HcalPoint_fZ[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Int_t           HcalPoint_fPdgCode[kMaxcbmroot_Hcal_HcalPoint];   //[cbmroot.Hcal.HcalPoint_]
   Int_t           HcalPointLite_;
   UInt_t          HcalPointLite_fUniqueID[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   UInt_t          HcalPointLite_fBits[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   TRef            HcalPointLite_fRefToLinks[kMaxcbmroot_HcalLite_HcalPointLite];
   Int_t           HcalPointLite_fTrackID[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   UInt_t          HcalPointLite_fEventId[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fPx[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fPy[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fPz[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fTime[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fLength[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fELoss[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Int_t           HcalPointLite_fDetectorID[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fX[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fY[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Double32_t      HcalPointLite_fZ[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Int_t           HcalPointLite_fPdgCode[kMaxcbmroot_HcalLite_HcalPointLite];   //[cbmroot.HcalLite.HcalPointLite_]
   Int_t           muonPoint_;
   UInt_t          muonPoint_fUniqueID[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   UInt_t          muonPoint_fBits[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   TRef            muonPoint_fRefToLinks[kMaxcbmroot_muon_muonPoint];
   Int_t           muonPoint_fTrackID[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   UInt_t          muonPoint_fEventId[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fPx[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fPy[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fPz[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fTime[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fLength[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fELoss[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Int_t           muonPoint_fDetectorID[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fX[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fY[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Double32_t      muonPoint_fZ[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
   Int_t           muonPoint_fPdgCode[kMaxcbmroot_muon_muonPoint];   //[cbmroot.muon.muonPoint_]
//   FairMCEventHeader *MCEventHeader_;
   UInt_t          MCEventHeader_TNamed_fUniqueID;
   UInt_t          MCEventHeader_TNamed_fBits;
   TString         MCEventHeader_TNamed_fName;
   TString         MCEventHeader_TNamed_fTitle;
   UInt_t          MCEventHeader_fRunId;
   UInt_t          MCEventHeader_fEventId;
   Double32_t      MCEventHeader_fX;
   Double32_t      MCEventHeader_fY;
   Double32_t      MCEventHeader_fZ;
   Double32_t      MCEventHeader_fT;
   Double32_t      MCEventHeader_fB;
   Int_t           MCEventHeader_fNPrim;
   Bool_t          MCEventHeader_fIsSet;
   Double32_t      MCEventHeader_fRotX;
   Double32_t      MCEventHeader_fRotY;
   Double32_t      MCEventHeader_fRotZ;

   // List of branches
   TBranch        *b_cbmroot_Stack_MCTrack_;   //!
   TBranch        *b_MCTrack_fUniqueID;   //!
   TBranch        *b_MCTrack_fBits;   //!
   TBranch        *b_MCTrack_fPdgCode;   //!
   TBranch        *b_MCTrack_fMotherId;   //!
   TBranch        *b_MCTrack_fPx;   //!
   TBranch        *b_MCTrack_fPy;   //!
   TBranch        *b_MCTrack_fPz;   //!
   TBranch        *b_MCTrack_fStartX;   //!
   TBranch        *b_MCTrack_fStartY;   //!
   TBranch        *b_MCTrack_fStartZ;   //!
   TBranch        *b_MCTrack_fStartT;   //!
   TBranch        *b_MCTrack_fW;   //!
   TBranch        *b_MCTrack_fNPoints;   //!
   TBranch        *b_cbmroot_veto_vetoPoint_;   //!
   TBranch        *b_vetoPoint_fUniqueID;   //!
   TBranch        *b_vetoPoint_fBits;   //!
   TBranch        *b_vetoPoint_fRefToLinks;   //!
   TBranch        *b_vetoPoint_fTrackID;   //!
   TBranch        *b_vetoPoint_fEventId;   //!
   TBranch        *b_vetoPoint_fPx;   //!
   TBranch        *b_vetoPoint_fPy;   //!
   TBranch        *b_vetoPoint_fPz;   //!
   TBranch        *b_vetoPoint_fTime;   //!
   TBranch        *b_vetoPoint_fLength;   //!
   TBranch        *b_vetoPoint_fELoss;   //!
   TBranch        *b_vetoPoint_fDetectorID;   //!
   TBranch        *b_vetoPoint_fX;   //!
   TBranch        *b_vetoPoint_fY;   //!
   TBranch        *b_vetoPoint_fZ;   //!
   TBranch        *b_vetoPoint_fPdgCode;   //!
   TBranch        *b_cbmroot_MagneticSpectrometer_ShipRpcPoint_;   //!
   TBranch        *b_ShipRpcPoint_fUniqueID;   //!
   TBranch        *b_ShipRpcPoint_fBits;   //!
   TBranch        *b_ShipRpcPoint_fRefToLinks;   //!
   TBranch        *b_ShipRpcPoint_fTrackID;   //!
   TBranch        *b_ShipRpcPoint_fEventId;   //!
   TBranch        *b_ShipRpcPoint_fPx;   //!
   TBranch        *b_ShipRpcPoint_fPy;   //!
   TBranch        *b_ShipRpcPoint_fPz;   //!
   TBranch        *b_ShipRpcPoint_fTime;   //!
   TBranch        *b_ShipRpcPoint_fLength;   //!
   TBranch        *b_ShipRpcPoint_fELoss;   //!
   TBranch        *b_ShipRpcPoint_fDetectorID;   //!
   TBranch        *b_ShipRpcPoint_fX;   //!
   TBranch        *b_ShipRpcPoint_fY;   //!
   TBranch        *b_ShipRpcPoint_fZ;   //!
   TBranch        *b_ShipRpcPoint_fPdgCode;   //!
   TBranch        *b_cbmroot_Target_TargetPoint_;   //!
   TBranch        *b_TargetPoint_fUniqueID;   //!
   TBranch        *b_TargetPoint_fBits;   //!
   TBranch        *b_TargetPoint_fRefToLinks;   //!
   TBranch        *b_TargetPoint_fTrackID;   //!
   TBranch        *b_TargetPoint_fEventId;   //!
   TBranch        *b_TargetPoint_fPx;   //!
   TBranch        *b_TargetPoint_fPy;   //!
   TBranch        *b_TargetPoint_fPz;   //!
   TBranch        *b_TargetPoint_fTime;   //!
   TBranch        *b_TargetPoint_fLength;   //!
   TBranch        *b_TargetPoint_fELoss;   //!
   TBranch        *b_TargetPoint_fDetectorID;   //!
   TBranch        *b_TargetPoint_fX;   //!
   TBranch        *b_TargetPoint_fY;   //!
   TBranch        *b_TargetPoint_fZ;   //!
   TBranch        *b_TargetPoint_fPdgCode;   //!
   TBranch        *b_cbmroot_strawtubes_strawtubesPoint_;   //!
   TBranch        *b_strawtubesPoint_fUniqueID;   //!
   TBranch        *b_strawtubesPoint_fBits;   //!
   TBranch        *b_strawtubesPoint_fRefToLinks;   //!
   TBranch        *b_strawtubesPoint_fTrackID;   //!
   TBranch        *b_strawtubesPoint_fEventId;   //!
   TBranch        *b_strawtubesPoint_fPx;   //!
   TBranch        *b_strawtubesPoint_fPy;   //!
   TBranch        *b_strawtubesPoint_fPz;   //!
   TBranch        *b_strawtubesPoint_fTime;   //!
   TBranch        *b_strawtubesPoint_fLength;   //!
   TBranch        *b_strawtubesPoint_fELoss;   //!
   TBranch        *b_strawtubesPoint_fDetectorID;   //!
   TBranch        *b_strawtubesPoint_fX;   //!
   TBranch        *b_strawtubesPoint_fY;   //!
   TBranch        *b_strawtubesPoint_fZ;   //!
   TBranch        *b_strawtubesPoint_fPdgCode;   //!
   TBranch        *b_strawtubesPoint_fdist2Wire;   //!
   TBranch        *b_cbmroot_Ecal_EcalPoint_;   //!
   TBranch        *b_EcalPoint_fUniqueID;   //!
   TBranch        *b_EcalPoint_fBits;   //!
   TBranch        *b_EcalPoint_fRefToLinks;   //!
   TBranch        *b_EcalPoint_fTrackID;   //!
   TBranch        *b_EcalPoint_fEventId;   //!
   TBranch        *b_EcalPoint_fPx;   //!
   TBranch        *b_EcalPoint_fPy;   //!
   TBranch        *b_EcalPoint_fPz;   //!
   TBranch        *b_EcalPoint_fTime;   //!
   TBranch        *b_EcalPoint_fLength;   //!
   TBranch        *b_EcalPoint_fELoss;   //!
   TBranch        *b_EcalPoint_fDetectorID;   //!
   TBranch        *b_EcalPoint_fX;   //!
   TBranch        *b_EcalPoint_fY;   //!
   TBranch        *b_EcalPoint_fZ;   //!
   TBranch        *b_EcalPoint_fPdgCode;   //!
   TBranch        *b_cbmroot_EcalLite_EcalPointLite_;   //!
   TBranch        *b_EcalPointLite_fUniqueID;   //!
   TBranch        *b_EcalPointLite_fBits;   //!
   TBranch        *b_EcalPointLite_fRefToLinks;   //!
   TBranch        *b_EcalPointLite_fTrackID;   //!
   TBranch        *b_EcalPointLite_fEventId;   //!
   TBranch        *b_EcalPointLite_fPx;   //!
   TBranch        *b_EcalPointLite_fPy;   //!
   TBranch        *b_EcalPointLite_fPz;   //!
   TBranch        *b_EcalPointLite_fTime;   //!
   TBranch        *b_EcalPointLite_fLength;   //!
   TBranch        *b_EcalPointLite_fELoss;   //!
   TBranch        *b_EcalPointLite_fDetectorID;   //!
   TBranch        *b_EcalPointLite_fX;   //!
   TBranch        *b_EcalPointLite_fY;   //!
   TBranch        *b_EcalPointLite_fZ;   //!
   TBranch        *b_EcalPointLite_fPdgCode;   //!
   TBranch        *b_cbmroot_Hcal_HcalPoint_;   //!
   TBranch        *b_HcalPoint_fUniqueID;   //!
   TBranch        *b_HcalPoint_fBits;   //!
   TBranch        *b_HcalPoint_fRefToLinks;   //!
   TBranch        *b_HcalPoint_fTrackID;   //!
   TBranch        *b_HcalPoint_fEventId;   //!
   TBranch        *b_HcalPoint_fPx;   //!
   TBranch        *b_HcalPoint_fPy;   //!
   TBranch        *b_HcalPoint_fPz;   //!
   TBranch        *b_HcalPoint_fTime;   //!
   TBranch        *b_HcalPoint_fLength;   //!
   TBranch        *b_HcalPoint_fELoss;   //!
   TBranch        *b_HcalPoint_fDetectorID;   //!
   TBranch        *b_HcalPoint_fX;   //!
   TBranch        *b_HcalPoint_fY;   //!
   TBranch        *b_HcalPoint_fZ;   //!
   TBranch        *b_HcalPoint_fPdgCode;   //!
   TBranch        *b_cbmroot_HcalLite_HcalPointLite_;   //!
   TBranch        *b_HcalPointLite_fUniqueID;   //!
   TBranch        *b_HcalPointLite_fBits;   //!
   TBranch        *b_HcalPointLite_fRefToLinks;   //!
   TBranch        *b_HcalPointLite_fTrackID;   //!
   TBranch        *b_HcalPointLite_fEventId;   //!
   TBranch        *b_HcalPointLite_fPx;   //!
   TBranch        *b_HcalPointLite_fPy;   //!
   TBranch        *b_HcalPointLite_fPz;   //!
   TBranch        *b_HcalPointLite_fTime;   //!
   TBranch        *b_HcalPointLite_fLength;   //!
   TBranch        *b_HcalPointLite_fELoss;   //!
   TBranch        *b_HcalPointLite_fDetectorID;   //!
   TBranch        *b_HcalPointLite_fX;   //!
   TBranch        *b_HcalPointLite_fY;   //!
   TBranch        *b_HcalPointLite_fZ;   //!
   TBranch        *b_HcalPointLite_fPdgCode;   //!
   TBranch        *b_cbmroot_muon_muonPoint_;   //!
   TBranch        *b_muonPoint_fUniqueID;   //!
   TBranch        *b_muonPoint_fBits;   //!
   TBranch        *b_muonPoint_fRefToLinks;   //!
   TBranch        *b_muonPoint_fTrackID;   //!
   TBranch        *b_muonPoint_fEventId;   //!
   TBranch        *b_muonPoint_fPx;   //!
   TBranch        *b_muonPoint_fPy;   //!
   TBranch        *b_muonPoint_fPz;   //!
   TBranch        *b_muonPoint_fTime;   //!
   TBranch        *b_muonPoint_fLength;   //!
   TBranch        *b_muonPoint_fELoss;   //!
   TBranch        *b_muonPoint_fDetectorID;   //!
   TBranch        *b_muonPoint_fX;   //!
   TBranch        *b_muonPoint_fY;   //!
   TBranch        *b_muonPoint_fZ;   //!
   TBranch        *b_muonPoint_fPdgCode;   //!
   TBranch        *b_cbmroot_Event_MCEventHeader_;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_TNamed_fUniqueID;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_TNamed_fBits;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_TNamed_fName;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_TNamed_fTitle;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fRunId;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fEventId;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fX;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fY;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fZ;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fT;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fB;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fNPrim;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fIsSet;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fRotX;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fRotY;   //!
   TBranch        *b_MCEventHeader_cbmroot_Event_MCEventHeader_fRotZ;   //!

   selection(TTree *tree=0);
   virtual ~selection();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef selection_cxx
selection::selection(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("ship.10.0.Pythia8-TGeant4.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("ship.10.0.Pythia8-TGeant4.root");
      }
      f->GetObject("cbmsim",tree);

   }
   Init(tree);
}

selection::~selection()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t selection::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t selection::LoadTree(Long64_t entry)
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

void selection::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
//  MCEventHeader_ = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("MCTrack", &MCTrack_, &b_cbmroot_Stack_MCTrack_);
   fChain->SetBranchAddress("MCTrack.fUniqueID", MCTrack_fUniqueID, &b_MCTrack_fUniqueID);
   fChain->SetBranchAddress("MCTrack.fBits", MCTrack_fBits, &b_MCTrack_fBits);
   fChain->SetBranchAddress("MCTrack.fPdgCode", MCTrack_fPdgCode, &b_MCTrack_fPdgCode);
   fChain->SetBranchAddress("MCTrack.fMotherId", MCTrack_fMotherId, &b_MCTrack_fMotherId);
   fChain->SetBranchAddress("MCTrack.fPx", MCTrack_fPx, &b_MCTrack_fPx);
   fChain->SetBranchAddress("MCTrack.fPy", MCTrack_fPy, &b_MCTrack_fPy);
   fChain->SetBranchAddress("MCTrack.fPz", MCTrack_fPz, &b_MCTrack_fPz);
   fChain->SetBranchAddress("MCTrack.fStartX", MCTrack_fStartX, &b_MCTrack_fStartX);
   fChain->SetBranchAddress("MCTrack.fStartY", MCTrack_fStartY, &b_MCTrack_fStartY);
   fChain->SetBranchAddress("MCTrack.fStartZ", MCTrack_fStartZ, &b_MCTrack_fStartZ);
   fChain->SetBranchAddress("MCTrack.fStartT", MCTrack_fStartT, &b_MCTrack_fStartT);
   fChain->SetBranchAddress("MCTrack.fW", MCTrack_fW, &b_MCTrack_fW);
   fChain->SetBranchAddress("MCTrack.fNPoints", MCTrack_fNPoints, &b_MCTrack_fNPoints);
   fChain->SetBranchAddress("vetoPoint", &vetoPoint_, &b_cbmroot_veto_vetoPoint_);
   fChain->SetBranchAddress("vetoPoint.fUniqueID", vetoPoint_fUniqueID, &b_vetoPoint_fUniqueID);
   fChain->SetBranchAddress("vetoPoint.fBits", vetoPoint_fBits, &b_vetoPoint_fBits);
   fChain->SetBranchAddress("vetoPoint.fRefToLinks", vetoPoint_fRefToLinks, &b_vetoPoint_fRefToLinks);
   fChain->SetBranchAddress("vetoPoint.fTrackID", vetoPoint_fTrackID, &b_vetoPoint_fTrackID);
   fChain->SetBranchAddress("vetoPoint.fEventId", vetoPoint_fEventId, &b_vetoPoint_fEventId);
   fChain->SetBranchAddress("vetoPoint.fPx", vetoPoint_fPx, &b_vetoPoint_fPx);
   fChain->SetBranchAddress("vetoPoint.fPy", vetoPoint_fPy, &b_vetoPoint_fPy);
   fChain->SetBranchAddress("vetoPoint.fPz", vetoPoint_fPz, &b_vetoPoint_fPz);
   fChain->SetBranchAddress("vetoPoint.fTime", vetoPoint_fTime, &b_vetoPoint_fTime);
   fChain->SetBranchAddress("vetoPoint.fLength", vetoPoint_fLength, &b_vetoPoint_fLength);
   fChain->SetBranchAddress("vetoPoint.fELoss", vetoPoint_fELoss, &b_vetoPoint_fELoss);
   fChain->SetBranchAddress("vetoPoint.fDetectorID", vetoPoint_fDetectorID, &b_vetoPoint_fDetectorID);
   fChain->SetBranchAddress("vetoPoint.fX", vetoPoint_fX, &b_vetoPoint_fX);
   fChain->SetBranchAddress("vetoPoint.fY", vetoPoint_fY, &b_vetoPoint_fY);
   fChain->SetBranchAddress("vetoPoint.fZ", vetoPoint_fZ, &b_vetoPoint_fZ);
   fChain->SetBranchAddress("vetoPoint.fPdgCode", vetoPoint_fPdgCode, &b_vetoPoint_fPdgCode);
   fChain->SetBranchAddress("ShipRpcPoint", &ShipRpcPoint_, &b_cbmroot_MagneticSpectrometer_ShipRpcPoint_);
   fChain->SetBranchAddress("ShipRpcPoint.fUniqueID", ShipRpcPoint_fUniqueID, &b_ShipRpcPoint_fUniqueID);
   fChain->SetBranchAddress("ShipRpcPoint.fBits", ShipRpcPoint_fBits, &b_ShipRpcPoint_fBits);
   fChain->SetBranchAddress("ShipRpcPoint.fRefToLinks", ShipRpcPoint_fRefToLinks, &b_ShipRpcPoint_fRefToLinks);
   fChain->SetBranchAddress("ShipRpcPoint.fTrackID", ShipRpcPoint_fTrackID, &b_ShipRpcPoint_fTrackID);
   fChain->SetBranchAddress("ShipRpcPoint.fEventId", ShipRpcPoint_fEventId, &b_ShipRpcPoint_fEventId);
   fChain->SetBranchAddress("ShipRpcPoint.fPx", ShipRpcPoint_fPx, &b_ShipRpcPoint_fPx);
   fChain->SetBranchAddress("ShipRpcPoint.fPy", ShipRpcPoint_fPy, &b_ShipRpcPoint_fPy);
   fChain->SetBranchAddress("ShipRpcPoint.fPz", ShipRpcPoint_fPz, &b_ShipRpcPoint_fPz);
   fChain->SetBranchAddress("ShipRpcPoint.fTime", ShipRpcPoint_fTime, &b_ShipRpcPoint_fTime);
   fChain->SetBranchAddress("ShipRpcPoint.fLength", ShipRpcPoint_fLength, &b_ShipRpcPoint_fLength);
   fChain->SetBranchAddress("ShipRpcPoint.fELoss", ShipRpcPoint_fELoss, &b_ShipRpcPoint_fELoss);
   fChain->SetBranchAddress("ShipRpcPoint.fDetectorID", ShipRpcPoint_fDetectorID, &b_ShipRpcPoint_fDetectorID);
   fChain->SetBranchAddress("ShipRpcPoint.fX", ShipRpcPoint_fX, &b_ShipRpcPoint_fX);
   fChain->SetBranchAddress("ShipRpcPoint.fY", ShipRpcPoint_fY, &b_ShipRpcPoint_fY);
   fChain->SetBranchAddress("ShipRpcPoint.fZ", ShipRpcPoint_fZ, &b_ShipRpcPoint_fZ);
   fChain->SetBranchAddress("ShipRpcPoint.fPdgCode", ShipRpcPoint_fPdgCode, &b_ShipRpcPoint_fPdgCode);
   fChain->SetBranchAddress("TargetPoint", &TargetPoint_, &b_cbmroot_Target_TargetPoint_);
   fChain->SetBranchAddress("TargetPoint.fUniqueID", TargetPoint_fUniqueID, &b_TargetPoint_fUniqueID);
   fChain->SetBranchAddress("TargetPoint.fBits", TargetPoint_fBits, &b_TargetPoint_fBits);
   fChain->SetBranchAddress("TargetPoint.fRefToLinks", TargetPoint_fRefToLinks, &b_TargetPoint_fRefToLinks);
   fChain->SetBranchAddress("TargetPoint.fTrackID", TargetPoint_fTrackID, &b_TargetPoint_fTrackID);
   fChain->SetBranchAddress("TargetPoint.fEventId", TargetPoint_fEventId, &b_TargetPoint_fEventId);
   fChain->SetBranchAddress("TargetPoint.fPx", TargetPoint_fPx, &b_TargetPoint_fPx);
   fChain->SetBranchAddress("TargetPoint.fPy", TargetPoint_fPy, &b_TargetPoint_fPy);
   fChain->SetBranchAddress("TargetPoint.fPz", TargetPoint_fPz, &b_TargetPoint_fPz);
   fChain->SetBranchAddress("TargetPoint.fTime", TargetPoint_fTime, &b_TargetPoint_fTime);
   fChain->SetBranchAddress("TargetPoint.fLength", TargetPoint_fLength, &b_TargetPoint_fLength);
   fChain->SetBranchAddress("TargetPoint.fELoss", TargetPoint_fELoss, &b_TargetPoint_fELoss);
   fChain->SetBranchAddress("TargetPoint.fDetectorID", TargetPoint_fDetectorID, &b_TargetPoint_fDetectorID);
   fChain->SetBranchAddress("TargetPoint.fX", TargetPoint_fX, &b_TargetPoint_fX);
   fChain->SetBranchAddress("TargetPoint.fY", TargetPoint_fY, &b_TargetPoint_fY);
   fChain->SetBranchAddress("TargetPoint.fZ", TargetPoint_fZ, &b_TargetPoint_fZ);
   fChain->SetBranchAddress("TargetPoint.fPdgCode", TargetPoint_fPdgCode, &b_TargetPoint_fPdgCode);
   fChain->SetBranchAddress("strawtubesPoint", &strawtubesPoint_, &b_cbmroot_strawtubes_strawtubesPoint_);
   fChain->SetBranchAddress("strawtubesPoint.fUniqueID", strawtubesPoint_fUniqueID, &b_strawtubesPoint_fUniqueID);
   fChain->SetBranchAddress("strawtubesPoint.fBits", strawtubesPoint_fBits, &b_strawtubesPoint_fBits);
   fChain->SetBranchAddress("strawtubesPoint.fRefToLinks", strawtubesPoint_fRefToLinks, &b_strawtubesPoint_fRefToLinks);
   fChain->SetBranchAddress("strawtubesPoint.fTrackID", strawtubesPoint_fTrackID, &b_strawtubesPoint_fTrackID);
   fChain->SetBranchAddress("strawtubesPoint.fEventId", strawtubesPoint_fEventId, &b_strawtubesPoint_fEventId);
   fChain->SetBranchAddress("strawtubesPoint.fPx", strawtubesPoint_fPx, &b_strawtubesPoint_fPx);
   fChain->SetBranchAddress("strawtubesPoint.fPy", strawtubesPoint_fPy, &b_strawtubesPoint_fPy);
   fChain->SetBranchAddress("strawtubesPoint.fPz", strawtubesPoint_fPz, &b_strawtubesPoint_fPz);
   fChain->SetBranchAddress("strawtubesPoint.fTime", strawtubesPoint_fTime, &b_strawtubesPoint_fTime);
   fChain->SetBranchAddress("strawtubesPoint.fLength", strawtubesPoint_fLength, &b_strawtubesPoint_fLength);
   fChain->SetBranchAddress("strawtubesPoint.fELoss", strawtubesPoint_fELoss, &b_strawtubesPoint_fELoss);
   fChain->SetBranchAddress("strawtubesPoint.fDetectorID", strawtubesPoint_fDetectorID, &b_strawtubesPoint_fDetectorID);
   fChain->SetBranchAddress("strawtubesPoint.fX", strawtubesPoint_fX, &b_strawtubesPoint_fX);
   fChain->SetBranchAddress("strawtubesPoint.fY", strawtubesPoint_fY, &b_strawtubesPoint_fY);
   fChain->SetBranchAddress("strawtubesPoint.fZ", strawtubesPoint_fZ, &b_strawtubesPoint_fZ);
   fChain->SetBranchAddress("strawtubesPoint.fPdgCode", strawtubesPoint_fPdgCode, &b_strawtubesPoint_fPdgCode);
   fChain->SetBranchAddress("strawtubesPoint.fdist2Wire", strawtubesPoint_fdist2Wire, &b_strawtubesPoint_fdist2Wire);
   fChain->SetBranchAddress("EcalPoint", &EcalPoint_, &b_cbmroot_Ecal_EcalPoint_);
   fChain->SetBranchAddress("EcalPoint.fUniqueID", EcalPoint_fUniqueID, &b_EcalPoint_fUniqueID);
   fChain->SetBranchAddress("EcalPoint.fBits", EcalPoint_fBits, &b_EcalPoint_fBits);
   fChain->SetBranchAddress("EcalPoint.fRefToLinks", EcalPoint_fRefToLinks, &b_EcalPoint_fRefToLinks);
   fChain->SetBranchAddress("EcalPoint.fTrackID", EcalPoint_fTrackID, &b_EcalPoint_fTrackID);
   fChain->SetBranchAddress("EcalPoint.fEventId", EcalPoint_fEventId, &b_EcalPoint_fEventId);
   fChain->SetBranchAddress("EcalPoint.fPx", EcalPoint_fPx, &b_EcalPoint_fPx);
   fChain->SetBranchAddress("EcalPoint.fPy", EcalPoint_fPy, &b_EcalPoint_fPy);
   fChain->SetBranchAddress("EcalPoint.fPz", EcalPoint_fPz, &b_EcalPoint_fPz);
   fChain->SetBranchAddress("EcalPoint.fTime", EcalPoint_fTime, &b_EcalPoint_fTime);
   fChain->SetBranchAddress("EcalPoint.fLength", EcalPoint_fLength, &b_EcalPoint_fLength);
   fChain->SetBranchAddress("EcalPoint.fELoss", EcalPoint_fELoss, &b_EcalPoint_fELoss);
   fChain->SetBranchAddress("EcalPoint.fDetectorID", EcalPoint_fDetectorID, &b_EcalPoint_fDetectorID);
   fChain->SetBranchAddress("EcalPoint.fX", EcalPoint_fX, &b_EcalPoint_fX);
   fChain->SetBranchAddress("EcalPoint.fY", EcalPoint_fY, &b_EcalPoint_fY);
   fChain->SetBranchAddress("EcalPoint.fZ", EcalPoint_fZ, &b_EcalPoint_fZ);
   fChain->SetBranchAddress("EcalPoint.fPdgCode", EcalPoint_fPdgCode, &b_EcalPoint_fPdgCode);
   fChain->SetBranchAddress("EcalPointLite", &EcalPointLite_, &b_cbmroot_EcalLite_EcalPointLite_);
   fChain->SetBranchAddress("EcalPointLite.fUniqueID", EcalPointLite_fUniqueID, &b_EcalPointLite_fUniqueID);
   fChain->SetBranchAddress("EcalPointLite.fBits", EcalPointLite_fBits, &b_EcalPointLite_fBits);
   fChain->SetBranchAddress("EcalPointLite.fRefToLinks", EcalPointLite_fRefToLinks, &b_EcalPointLite_fRefToLinks);
   fChain->SetBranchAddress("EcalPointLite.fTrackID", EcalPointLite_fTrackID, &b_EcalPointLite_fTrackID);
   fChain->SetBranchAddress("EcalPointLite.fEventId", EcalPointLite_fEventId, &b_EcalPointLite_fEventId);
   fChain->SetBranchAddress("EcalPointLite.fPx", EcalPointLite_fPx, &b_EcalPointLite_fPx);
   fChain->SetBranchAddress("EcalPointLite.fPy", EcalPointLite_fPy, &b_EcalPointLite_fPy);
   fChain->SetBranchAddress("EcalPointLite.fPz", EcalPointLite_fPz, &b_EcalPointLite_fPz);
   fChain->SetBranchAddress("EcalPointLite.fTime", EcalPointLite_fTime, &b_EcalPointLite_fTime);
   fChain->SetBranchAddress("EcalPointLite.fLength", EcalPointLite_fLength, &b_EcalPointLite_fLength);
   fChain->SetBranchAddress("EcalPointLite.fELoss", EcalPointLite_fELoss, &b_EcalPointLite_fELoss);
   fChain->SetBranchAddress("EcalPointLite.fDetectorID", EcalPointLite_fDetectorID, &b_EcalPointLite_fDetectorID);
   fChain->SetBranchAddress("EcalPointLite.fX", EcalPointLite_fX, &b_EcalPointLite_fX);
   fChain->SetBranchAddress("EcalPointLite.fY", EcalPointLite_fY, &b_EcalPointLite_fY);
   fChain->SetBranchAddress("EcalPointLite.fZ", EcalPointLite_fZ, &b_EcalPointLite_fZ);
   fChain->SetBranchAddress("EcalPointLite.fPdgCode", EcalPointLite_fPdgCode, &b_EcalPointLite_fPdgCode);
   fChain->SetBranchAddress("HcalPoint", &HcalPoint_, &b_cbmroot_Hcal_HcalPoint_);
   fChain->SetBranchAddress("HcalPoint.fUniqueID", HcalPoint_fUniqueID, &b_HcalPoint_fUniqueID);
   fChain->SetBranchAddress("HcalPoint.fBits", HcalPoint_fBits, &b_HcalPoint_fBits);
   fChain->SetBranchAddress("HcalPoint.fRefToLinks", HcalPoint_fRefToLinks, &b_HcalPoint_fRefToLinks);
   fChain->SetBranchAddress("HcalPoint.fTrackID", HcalPoint_fTrackID, &b_HcalPoint_fTrackID);
   fChain->SetBranchAddress("HcalPoint.fEventId", HcalPoint_fEventId, &b_HcalPoint_fEventId);
   fChain->SetBranchAddress("HcalPoint.fPx", HcalPoint_fPx, &b_HcalPoint_fPx);
   fChain->SetBranchAddress("HcalPoint.fPy", HcalPoint_fPy, &b_HcalPoint_fPy);
   fChain->SetBranchAddress("HcalPoint.fPz", HcalPoint_fPz, &b_HcalPoint_fPz);
   fChain->SetBranchAddress("HcalPoint.fTime", HcalPoint_fTime, &b_HcalPoint_fTime);
   fChain->SetBranchAddress("HcalPoint.fLength", HcalPoint_fLength, &b_HcalPoint_fLength);
   fChain->SetBranchAddress("HcalPoint.fELoss", HcalPoint_fELoss, &b_HcalPoint_fELoss);
   fChain->SetBranchAddress("HcalPoint.fDetectorID", HcalPoint_fDetectorID, &b_HcalPoint_fDetectorID);
   fChain->SetBranchAddress("HcalPoint.fX", HcalPoint_fX, &b_HcalPoint_fX);
   fChain->SetBranchAddress("HcalPoint.fY", HcalPoint_fY, &b_HcalPoint_fY);
   fChain->SetBranchAddress("HcalPoint.fZ", HcalPoint_fZ, &b_HcalPoint_fZ);
   fChain->SetBranchAddress("HcalPoint.fPdgCode", HcalPoint_fPdgCode, &b_HcalPoint_fPdgCode);
   fChain->SetBranchAddress("HcalPointLite", &HcalPointLite_, &b_cbmroot_HcalLite_HcalPointLite_);
   fChain->SetBranchAddress("HcalPointLite.fUniqueID", HcalPointLite_fUniqueID, &b_HcalPointLite_fUniqueID);
   fChain->SetBranchAddress("HcalPointLite.fBits", HcalPointLite_fBits, &b_HcalPointLite_fBits);
   fChain->SetBranchAddress("HcalPointLite.fRefToLinks", HcalPointLite_fRefToLinks, &b_HcalPointLite_fRefToLinks);
   fChain->SetBranchAddress("HcalPointLite.fTrackID", HcalPointLite_fTrackID, &b_HcalPointLite_fTrackID);
   fChain->SetBranchAddress("HcalPointLite.fEventId", HcalPointLite_fEventId, &b_HcalPointLite_fEventId);
   fChain->SetBranchAddress("HcalPointLite.fPx", HcalPointLite_fPx, &b_HcalPointLite_fPx);
   fChain->SetBranchAddress("HcalPointLite.fPy", HcalPointLite_fPy, &b_HcalPointLite_fPy);
   fChain->SetBranchAddress("HcalPointLite.fPz", HcalPointLite_fPz, &b_HcalPointLite_fPz);
   fChain->SetBranchAddress("HcalPointLite.fTime", HcalPointLite_fTime, &b_HcalPointLite_fTime);
   fChain->SetBranchAddress("HcalPointLite.fLength", HcalPointLite_fLength, &b_HcalPointLite_fLength);
   fChain->SetBranchAddress("HcalPointLite.fELoss", HcalPointLite_fELoss, &b_HcalPointLite_fELoss);
   fChain->SetBranchAddress("HcalPointLite.fDetectorID", HcalPointLite_fDetectorID, &b_HcalPointLite_fDetectorID);
   fChain->SetBranchAddress("HcalPointLite.fX", HcalPointLite_fX, &b_HcalPointLite_fX);
   fChain->SetBranchAddress("HcalPointLite.fY", HcalPointLite_fY, &b_HcalPointLite_fY);
   fChain->SetBranchAddress("HcalPointLite.fZ", HcalPointLite_fZ, &b_HcalPointLite_fZ);
   fChain->SetBranchAddress("HcalPointLite.fPdgCode", HcalPointLite_fPdgCode, &b_HcalPointLite_fPdgCode);
   fChain->SetBranchAddress("muonPoint", &muonPoint_, &b_cbmroot_muon_muonPoint_);
   fChain->SetBranchAddress("muonPoint.fUniqueID", muonPoint_fUniqueID, &b_muonPoint_fUniqueID);
   fChain->SetBranchAddress("muonPoint.fBits", muonPoint_fBits, &b_muonPoint_fBits);
   fChain->SetBranchAddress("muonPoint.fRefToLinks", muonPoint_fRefToLinks, &b_muonPoint_fRefToLinks);
   fChain->SetBranchAddress("muonPoint.fTrackID", muonPoint_fTrackID, &b_muonPoint_fTrackID);
   fChain->SetBranchAddress("muonPoint.fEventId", muonPoint_fEventId, &b_muonPoint_fEventId);
   fChain->SetBranchAddress("muonPoint.fPx", muonPoint_fPx, &b_muonPoint_fPx);
   fChain->SetBranchAddress("muonPoint.fPy", muonPoint_fPy, &b_muonPoint_fPy);
   fChain->SetBranchAddress("muonPoint.fPz", muonPoint_fPz, &b_muonPoint_fPz);
   fChain->SetBranchAddress("muonPoint.fTime", muonPoint_fTime, &b_muonPoint_fTime);
   fChain->SetBranchAddress("muonPoint.fLength", muonPoint_fLength, &b_muonPoint_fLength);
   fChain->SetBranchAddress("muonPoint.fELoss", muonPoint_fELoss, &b_muonPoint_fELoss);
   fChain->SetBranchAddress("muonPoint.fDetectorID", muonPoint_fDetectorID, &b_muonPoint_fDetectorID);
   fChain->SetBranchAddress("muonPoint.fX", muonPoint_fX, &b_muonPoint_fX);
   fChain->SetBranchAddress("muonPoint.fY", muonPoint_fY, &b_muonPoint_fY);
   fChain->SetBranchAddress("muonPoint.fZ", muonPoint_fZ, &b_muonPoint_fZ);
   fChain->SetBranchAddress("muonPoint.fPdgCode", muonPoint_fPdgCode, &b_muonPoint_fPdgCode);
//   fChain->SetBranchAddress("MCEventHeader.", &MCEventHeader_, &b_cbmroot_Event_MCEventHeader_);
   fChain->SetBranchAddress("MCEventHeader.TNamed.fUniqueID", &MCEventHeader_TNamed_fUniqueID, &b_MCEventHeader_cbmroot_Event_MCEventHeader_TNamed_fUniqueID);
   fChain->SetBranchAddress("MCEventHeader.TNamed.fBits", &MCEventHeader_TNamed_fBits, &b_MCEventHeader_cbmroot_Event_MCEventHeader_TNamed_fBits);
   fChain->SetBranchAddress("MCEventHeader.TNamed.fName", &MCEventHeader_TNamed_fName, &b_MCEventHeader_cbmroot_Event_MCEventHeader_TNamed_fName);
   fChain->SetBranchAddress("MCEventHeader.TNamed.fTitle", &MCEventHeader_TNamed_fTitle, &b_MCEventHeader_cbmroot_Event_MCEventHeader_TNamed_fTitle);
   fChain->SetBranchAddress("MCEventHeader.fRunId", &MCEventHeader_fRunId, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fRunId);
   fChain->SetBranchAddress("MCEventHeader.fEventId", &MCEventHeader_fEventId, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fEventId);
   fChain->SetBranchAddress("MCEventHeader.fX", &MCEventHeader_fX, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fX);
   fChain->SetBranchAddress("MCEventHeader.fY", &MCEventHeader_fY, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fY);
   fChain->SetBranchAddress("MCEventHeader.fZ", &MCEventHeader_fZ, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fZ);
   fChain->SetBranchAddress("MCEventHeader.fT", &MCEventHeader_fT, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fT);
   fChain->SetBranchAddress("MCEventHeader.fB", &MCEventHeader_fB, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fB);
   fChain->SetBranchAddress("MCEventHeader.fNPrim", &MCEventHeader_fNPrim, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fNPrim);
   fChain->SetBranchAddress("MCEventHeader.fIsSet", &MCEventHeader_fIsSet, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fIsSet);
   fChain->SetBranchAddress("MCEventHeader.fRotX", &MCEventHeader_fRotX, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fRotX);
   fChain->SetBranchAddress("MCEventHeader.fRotY", &MCEventHeader_fRotY, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fRotY);
   fChain->SetBranchAddress("MCEventHeader.fRotZ", &MCEventHeader_fRotZ, &b_MCEventHeader_cbmroot_Event_MCEventHeader_fRotZ);
   Notify();
}

Bool_t selection::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void selection::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t selection::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef selection_cxx
