#include "HWWAnalysisCode/HWW_WHFalpha1lowpt.h"
#include <limits>

// uncomment the following line to enable debug printouts
// #define _DEBUG_
// you can perform debug printouts with statements like this
// DEBUG("error number %d occurred",someInteger);

// be careful to not move the _DEBUG_ flag behind the following line
// otherwise, it will show no effect
#include "QFramework/TQLibrary.h"
using namespace xAOD;
ClassImp(HWW_WHFalpha1lowpt)

//______________________________________________________________________________________________

HWW_WHFalpha1lowpt::HWW_WHFalpha1lowpt() :
  fDoTruth(false)
{
  // default constructor
  DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________

HWW_WHFalpha1lowpt::~HWW_WHFalpha1lowpt(){
  // default destructor
  DEBUGclass("destructor called");
}

#define XAOD_STANDALONE 1
// put here any EDM includes you might need, e.g.
//#include "xAODParticleEvent/CompositeParticleContainer.h"

//______________________________________________________________________________________________

bool HWW_WHFalpha1lowpt::initializeSelf(){

  TString CandName = "";
  if(!this->fSample->getTagString("~cand",CandName)) return false;
  this->mCandName = (this->fDoTruth ? "EventTruth" : "Event")+CandName;

  return true;
}

//______________________________________________________________________________________________

double HWW_WHFalpha1lowpt::getValue() const {
  // value retrieval function, called on every event for every cut and histogram
  DEBUGclass("entering function");

  // the TQEventObservable only works in ASG_RELEASE RootCore, hence
  // we encapsulate the implementation in an ifdef/ifndef block
  #ifndef HAS_XAOD
  #warning "using plain ROOT compilation scheme - please add '-DASG_RELEASE' to your packages 'Makefile.RootCore'"
  return std::numeric_limits<double>::quiet_NaN();
  #else 
  // in the rest of this function, you should retrieve the data and calculate your return value
  // here is the place where most of your custom code should go
  // a couple of comments should guide you through the process
  // when writing your code, please keep in mind that this code can be executed several times on every event
  // make your code efficient. catch all possible problems. when in doubt, contact experts!
  
  // first, you can retrieve the data members you needwith the 'retrieve' method
  // as arguments to this function, pass the member pointer to the container as well as the container name
  // example block:

  //  xAOD::CompositeParticleContainer* cont = NULL;
  //this->retrieve(cont, "EventWH");
  //DEBUGclass("cont->size() %i",cont->size());
  //if(cont->size()==0) return -1;

  if(!this->fEvent->retrieve(this->mCand, this->mCandName.Data()).isSuccess()){
    DEBUGclass("failed to retrieve candidates!");
    return 0;
  }

  DEBUGclass("retrieve candidates! %s",this->mCandName.Data());

  DEBUGclass(" this->mCand->size() %i",this->mCand->size());
  if(this->mCand->size()==0) return -1;

  float retval =-999.;

  const xAOD::CompositeParticle *Evt = this->mCand->at(0);
  const xAOD::IParticle *lep0 = Evt->part(0);
  const xAOD::IParticle *lep1 = Evt->part(1);
  const xAOD::IParticle *lep2 = Evt->part(2);

  const MissingET *METCand = Evt->missingET();
  float METx = METCand->mpx();
  float METy = METCand->mpy();
  const IParticle *l1 = Evt->part(0);
  const IParticle *l2 = Evt->part(1);
  const IParticle *l3 = Evt->part(2);

  float lepPt0 = (lep0->p4()).Pt();
  float lepPt1 = (lep1->p4()).Pt();
  float lepPt2 = (lep2->p4()).Pt();

  if(lepPt0<=lepPt1 && lepPt0<=lepPt2){
    l1 = lep0;
    if(lepPt1<=lepPt2){
      if(lep0->auxdata<float>("charge")*lep1->auxdata<float>("charge")<0){
        l2 = lep1; l3 = lep2;
      } else{l2 = lep2; l3 = lep1;}
    }
    else if(lep0->auxdata<float>("charge")*lep2->auxdata<float>("charge")<0){
      l2 = lep2; l3 = lep1;
    } else{l2 = lep1; l3 = lep2;}
  }
  else if(lepPt1<=lepPt0 && lepPt1<=lepPt2){
    l1 = lep1;
    if(lepPt0<=lepPt2){
      if(lep1->auxdata<float>("charge")*lep0->auxdata<float>("charge")<0){
        l2 = lep0; l3 = lep2;
      } else{l2 = lep2; l3 = lep0;}
    }
    else if(lep1->auxdata<float>("charge")*lep2->auxdata<float>("charge")<0){
      l2 = lep2; l3 = lep0;
    } else{l2 = lep0; l3 = lep2;}
  }
  else if(lepPt2<=lepPt0 && lepPt2<=lepPt1){
    l1 = lep2;
    if(lepPt0<=lepPt1){
      if(lep2->auxdata<float>("charge")*lep0->auxdata<float>("charge")<0){
        l2 = lep0; l3 = lep1;
      } else{l2 = lep1; l3 = lep0;}
    }
    else if(lep2->auxdata<float>("charge")*lep1->auxdata<float>("charge")<0){
      l2 = lep1; l3 = lep0;
    } else{l2 = lep0; l3 = lep1;}
  }

  // calculate Falpha1 ------------------------------------------------------------------------------

  int durchlaeufe = 100; //trials
  double schrittfolge = 0.01; //stepsize
  double zaehler = 0; // successful runs
  double binWeight[10] = {0.000523209,0.0185422,0.0773298,0.1551565,0.19408925,0.1874,0.15113275,0.12033775,0.0721813,0.02330715};

  double a1 = 91187.6*91187.6;
  double b1 = 1776.82*1776.82;
  double d1;
  double e1 = (l2->p4()).E()*(l2->p4()).E();
  double a2 = (80385.0*80385.0) / 2;
  double b2 = (l3->p4()).E();
  double e2 = -(l3->p4()).Pz();
  for(int i=1;i<=durchlaeufe;i++){
    double alpha1 = 1.0/(i*schrittfolge); // alpha1 > 1 (!)
    double c1 = alpha1*(l1->p4()).E()*(l2->p4()).E();
    if((alpha1*alpha1*(l1->p4()).E()*(l1->p4()).E()-b1)>=0){
      d1 = sqrt(alpha1*alpha1*(l1->p4()).E()*(l1->p4()).E()-b1)*cos((l1->p4()).Angle((l2->p4()).Vect()));
    } else {continue;} // Abbruch (!)

    // calculate alpha2
    double f1 = a1*a1*d1*d1*e1 - 4*a1*b1*d1*d1*e1 + 4*b1*b1*d1*d1*e1 - 4*b1*c1*c1*d1*d1 + 4*b1*d1*d1*d1*d1*e1;
    double g1 = c1*c1 - d1*d1*e1;
    double alpha2_1;
    double alpha2_2;
    if(f1>=0 && g1!=0){
      alpha2_1 = (sqrt(f1) + a1*c1 - 2*b1*c1) / (2*g1);
      alpha2_2 = (- sqrt(f1) + a1*c1 - 2*b1*c1) / (2*g1);
    } else {continue;} // Abbruch (!)

    if(alpha2_1 < 1 && alpha2_2 < 1){continue;} // Abbruch (!)

    // calculate Px(nu_W), Py(nu_W) via METx, METy
    if(alpha2_1 > 1){
      double nuPx_1 = METx - (alpha1-1)*(l1->p4()).E()*sin((l1->p4()).Theta())*cos((l1->p4()).Phi()) - (alpha2_1-1)*(l2->p4()).E()*sin((l2->p4()).Theta())*cos((l2->p4()).Phi());
      double nuPy_1 = METy - (alpha1-1)*(l1->p4()).E()*sin((l1->p4()).Theta())*sin((l1->p4()).Phi()) - (alpha2_1-1)*(l2->p4()).E()*sin((l2->p4()).Theta())*sin((l2->p4()).Phi());

      // calculate Pz(nu_W) via m_Z
      double c2_1 = nuPx_1*nuPx_1 + nuPy_1*nuPy_1;
      double d2_1 = - (l3->p4()).Px()*nuPx_1 - (l3->p4()).Py()*nuPy_1;

      double f2_1 = a2*a2*b2*b2 - 2*a2*b2*b2*d2_1 - b2*b2*b2*b2*c2_1 + b2*b2*c2_1*e2*e2 + b2*b2*d2_1*d2_1;
      double g2_1 = b2*b2 - e2*e2;

      if(f2_1>=0 && g2_1!=0){
        //double nuPz_1 = (sqrt(f2_1) - a2*e2 + d2_1*e2) / g2_1;
        //double nuPz_2 = (- sqrt(f2_1) - a2*e2 + d2_1*e2) / g2_1;
        //zaehler = zaehler + binWeight[(i-1)/10]*binWeight[(int)((1.0/alpha2_1)*10.0)]; // successful run (!!!)
        zaehler = zaehler + binWeight[(i-1)/10];
        //zaehler = zaehler + 1;
      }
    }
    if(alpha2_2 > 1){
      double nuPx_2 = METx - (alpha1-1)*(l1->p4()).E()*sin((l1->p4()).Theta())*cos((l1->p4()).Phi()) - (alpha2_2-1)*(l2->p4()).E()*sin((l2->p4()).Theta())*cos((l2->p4()).Phi());
      double nuPy_2 = METy - (alpha1-1)*(l1->p4()).E()*sin((l1->p4()).Theta())*sin((l1->p4()).Phi()) - (alpha2_2-1)*(l2->p4()).E()*sin((l2->p4()).Theta())*sin((l2->p4()).Phi());

      // calculate Pz(nu_W) via m_Z
      double c2_2 = nuPx_2*nuPx_2 + nuPy_2*nuPy_2;
      double d2_2 = - (l3->p4()).Px()*nuPx_2 - (l3->p4()).Py()*nuPy_2;

      double f2_2 = a2*a2*b2*b2 - 2*a2*b2*b2*d2_2 - b2*b2*b2*b2*c2_2 + b2*b2*c2_2*e2*e2 + b2*b2*d2_2*d2_2;
      double g2_2 = b2*b2 - e2*e2;

      if(f2_2>=0 && g2_2!=0){
        //double nuPz_3 = (sqrt(f2_2) - a2*e2 + d2_2*e2) / g2_2;
        //double nuPz_4 = (- sqrt(f2_2) - a2*e2 + d2_2*e2) / g2_2;
        //zaehler = zaehler + binWeight[(i-1)/10]*binWeight[(int)((1.0/alpha2_2)*10.0)]; // successful run (!!!)
        zaehler = zaehler + binWeight[(i-1)/10];
        //zaehler = zaehler + 1;
      }
    }
  } // end of loop over alpha1

  double coRRector = 0; for(int coR=0;coR<10;coR++){coRRector = coRRector + 10*binWeight[coR];}

  retval = zaehler / (2*coRRector);

  DEBUGclass("returning");
  return retval;
  #endif
}

//______________________________________________________________________________________________

HWW_WHFalpha1lowpt::HWW_WHFalpha1lowpt(const TString& expression, const bool doTruth):
  TQEventObservable(expression),
  fDoTruth(doTruth)
{
  // constructor with expression argument
  DEBUGclass("constructor called with '%s'",expression.Data());
  // the predefined string member "expression" allows your observable to store an expression of your choice
  // this string will be the verbatim argument you passed to the constructor of your observable
  // you can use it to choose between different modes or pass configuration options to your observable
  this->SetName(TQObservable::makeObservableName(expression));
  this->setExpression(expression);
}

//______________________________________________________________________________________________

const TString& HWW_WHFalpha1lowpt::getExpression() const {
  // retrieve the expression associated with this observable
  return this->fExpression;
}

//______________________________________________________________________________________________

bool HWW_WHFalpha1lowpt::hasExpression() const {
  // check if this observable type knows expressions
  return true;
}

//______________________________________________________________________________________________

void HWW_WHFalpha1lowpt::setExpression(const TString& expr){
  // set the expression to a given string
  this->fExpression = expr;
}
