//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file pm_envelope_sg.cpp: tidal perturbation of polytropic stellar envelope by one point mass, monopole self gravity
//======================================================================================

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#define NARRAY 10000
#define NGRAV 200


// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"
#include "../utils/utils.hpp"
#include "../outputs/outputs.hpp"
#include "../scalars/scalars.hpp"




Real Interpolate1DArrayEven(Real *x,Real *y,Real x0, int length);

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
		 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt,  const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim,
		  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
		  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar); 


void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void ParticleAccelsPreInt(Real GMenv, Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void SumGasOnParticleAccels(Mesh *pm, Real (&xi)[3],Real (&ag1i)[3], Real (&ag2i)[3]);

void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]);

int RefinementCondition(MeshBlock *pmb);

void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]);

void WritePMTrackfile(Mesh *pm, ParameterInput *pin);

Real GetGM2factor(Real time);

void SumComPosVel(Mesh *pm, Real (&xi)[3], Real (&vi)[3],
		     Real (&xcomSum)[3],Real (&vcomSum)[3],
		     Real (&xcom_star)[3],Real (&vcom_star)[3],
		     Real &mg, Real &mg_star);

void SumTrackfileDiagnostics(Mesh *pm, Real (&xi)[3], Real (&vi)[3],
			     Real (&lp)[3],Real (&lg)[3],Real (&ldo)[3],
			     Real &EK, Real &EPot, Real &EI,Real &Edo,
			     Real &EK_star, Real &EPot_star, Real &EI_star,
			     Real &EK_ej, Real &EPot_ej, Real &EI_ej,
			     Real &M_star, Real &mr1, Real &mr12,
			     Real &mb, Real &mu,
			     Real &Eorb, Real &Lz_star, Real &Lz_orb, Real &Lz_ej);

void SumMencProfile(Mesh *pm, Real (&menc)[NGRAV]);

Real fspline(Real r, Real eps);
Real pspline(Real r, Real eps);

bool instar(Real den, Real r);


Real kappa(Real rho, Real T);

void updateGM2(Real sep);


// global (to this file) problem parameters
Real gamma_gas; 
Real da,pa; // ambient density, pressure
Real rho[NARRAY], p[NARRAY], rad[NARRAY], menc_init[NARRAY];  // initial profile
Real logr[NGRAV],menc[NGRAV]; // enclosed mass profile

Real GM2, GM1,GM2i; // point masses
Real rsoft2; // softening length of PM 2
Real t_relax,t_mass_on; // time to damp fluid motion, time to turn on M2 over
int  include_gas_backreaction, corotating_frame; // flags for output, gas backreaction on EOM, frame choice
int n_particle_substeps; // substepping of particle integration

Real xi[3], vi[3], agas1i[3], agas2i[3]; // cartesian positions/vels of the secondary object, gas->particle acceleration
Real xcom[3], vcom[3]; // cartesian pos/vel of the COM of the particle/gas system
Real xcom_star[3], vcom_star[3]; // cartesian pos/vel of the COM of the star
Real lp[3], lg[3], ldo[3];  // particle, gas, and rate of angular momentum loss
Real EK, EPot, EI, Edo, EK_star, EPot_star, EI_star, EK_ej, EPot_ej, EI_ej, M_star, mr1, mr12,mb,mu, Eorb, Lz_star, Lz_orb, Lz_ej; // diagnostic output

Real Omega[3],  Omega_envelope;  // vector rotation of the frame, initial envelope

Real trackfile_next_time, trackfile_dt;
int  trackfile_number;
int  mode;  // mode=1 (polytrope), mode=2 (wind BC) 

Real Ggrav;

Real separation_start,separation_stop_min, separation_stop_max; // particle separation to abort the integration.

int is_restart;

// Static Refinement with AMR Params
//Real x1_min_level1, x1_max_level1,x2_min_level1, x2_max_level1;
Real x1_min_derefine;

bool do_pre_integrate;
bool fixed_orbit;
Real Omega_orb_fixed,sma_fixed;

Real output_next_sep,dsep_output; // controling user forced output (set with dt=999.)

int update_grav_every;
bool inert_bg;  // should the background respond to forces
Real tau_relax_start,tau_relax_end;
Real rstar_initial,mstar_initial;


bool cooling; // whether to apply cooling function or not
Real Lstar; // stellar luminosity
Real mykappa;
Real fvir;

Real sigmaSB = 5.67051e-5; //erg / cm^2 / K^4
Real kB = 1.380658e-16; // erg / K
Real mH = 1.6733e-24; // g
Real X,Y,Z; // mass fractions composition

//int rotation_mode; // setting for rotation 1 = solid body, 2 = experimental, differential
bool diff_rot_exp;

bool update_gm2_sep; //change gm2 as a function of separation
Real dmin = 1.e99;

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{

  // read in some global params (to this file)
 
  // first non-mode-dependent settings
  pa   = pin->GetOrAddReal("problem","pamb",1.0);
  da   = pin->GetOrAddReal("problem","damb",1.0);
  gamma_gas = pin->GetReal("hydro","gamma");

  Ggrav = pin->GetOrAddReal("problem","Ggrav",6.67408e-8);
  GM2 = pin->GetOrAddReal("problem","GM2",0.0);
  //GM1 = pin->GetOrAddReal("problem","GM1",1.0);
  GM2i = GM2; // set initial GM2

  rsoft2 = pin->GetOrAddReal("problem","rsoft2",0.1);
  t_relax = pin->GetOrAddReal("problem","trelax",0.0);
  tau_relax_start = pin->GetOrAddReal("problem","tau_relax_start",1.0);
  tau_relax_end = pin->GetOrAddReal("problem","tau_relax_end",100.0);
  t_mass_on = pin->GetOrAddReal("problem","t_mass_on",0.0);
  corotating_frame = pin->GetInteger("problem","corotating_frame");

  trackfile_dt = pin->GetOrAddReal("problem","trackfile_dt",0.01);

  include_gas_backreaction = pin->GetInteger("problem","gas_backreaction");
  n_particle_substeps = pin->GetInteger("problem","n_particle_substeps");

  separation_stop_min = pin->GetOrAddReal("problem","separation_stop_min",0.0);
  separation_stop_max = pin->GetOrAddReal("problem","separation_stop_max",1.e99);
  separation_start = pin->GetOrAddReal("problem","separation_start",1.e99);

  // These are used for static refinement when using AMR
  x1_min_derefine = pin->GetOrAddReal("problem","x1_min_derefine",0.0);

  // fixed orbit parameters
  fixed_orbit = pin->GetOrAddBoolean("problem","fixed_orbit",false);
  Omega_orb_fixed = pin->GetOrAddReal("problem","omega_orb_fixed",0.5);

  // separation based ouput, triggered when dt=999. for an output type
  dsep_output = pin->GetOrAddReal("problem","dsep_output",1.0);
  Real output_next_sep_max = pin->GetOrAddReal("problem","output_next_sep_max",1.0);
  output_next_sep = output_next_sep_max;

  // gravity
  update_grav_every = pin->GetOrAddInteger("problem","update_grav_every",1);
  rstar_initial = pin->GetReal("problem","rstar_initial");  // FOR RESCALING OF STELLAR PROFILE
  mstar_initial = pin->GetReal("problem","mstar_initial");
  
  // background
  inert_bg = pin->GetOrAddBoolean("problem","inert_bg",false);

  // cooling parameters
  cooling = pin->GetOrAddBoolean("problem","cooling",false);
  Lstar = pin->GetOrAddReal("problem","lstar",4.e33);
  mykappa = pin->GetOrAddReal("problem","kappa",1e-3);
  fvir = pin->GetOrAddReal("problem","fvir",0.1);

  X = pin->GetOrAddReal("problem","X",0.7);
  Z = pin->GetOrAddReal("problem","Z",0.02);
  Y = 1.0 - X - Z;

  // rotation mode
  //rotation_mode = pin->GetOrAddInteger("problem","rotation_mode",1);
  //eps_rot = pin->GetOrAddReal("problem","eps_rot",0.0);
  diff_rot_exp = pin->GetOrAddReal("problem","diff_rot_exp",0.0);

  // gm2 decrease
  update_gm2_sep = pin->GetOrAddBoolean("problem","update_gm2_sep",false);
  

 

  // local vars
  Real rmin = pin->GetOrAddReal("mesh","x1min",0.0);
  Real rmax = pin->GetOrAddReal("mesh","x1max",0.0);
  Real thmin = pin->GetOrAddReal("mesh","x2min",0.0);
  Real thmax = pin->GetOrAddReal("mesh","x2max",0.0);

  Real sma = pin->GetOrAddReal("problem","sma",2.0);
  Real ecc = pin->GetOrAddReal("problem","ecc",0.0);
  Real fcorot = pin->GetOrAddReal("problem","fcorotation",0.0);
  Real Omega_orb, vcirc;
  

   // allocate MESH data for the particle pos/vel, Omega frame
  AllocateRealUserMeshDataField(5);
  ruser_mesh_data[0].NewAthenaArray(3);
  ruser_mesh_data[1].NewAthenaArray(3);
  ruser_mesh_data[2].NewAthenaArray(3);
  ruser_mesh_data[3].NewAthenaArray(3);
  ruser_mesh_data[4].NewAthenaArray(3);
  
  
  // enroll the BCs
  if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiodeOuterX1);
  }

  // Enroll a Source Function
  EnrollUserExplicitSourceFunction(TwoPointMass);

  // Enroll AMR
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  // Enroll extra history output
  //AllocateUserHistoryOutput(8);
  //EnrollUserHistoryOutput(0, mxOmegaEnv, "mxOmegaEnv");



  // Check the scalar count
  if(NSCALARS != 8){
    std::cout<<"COMPILED WITH "<<NSCALARS<<" SCALARS but 8 are required!!!";
  }

  // always write at startup
  trackfile_next_time = time;
  trackfile_number = 0;


    
  // read in profile arrays from file
  std::ifstream infile("hse_profile.dat"); 
  for(int i=0;i<NARRAY;i++){
    infile >> rad[i] >> rho[i] >> p[i] >> menc_init[i];
    //std:: cout << rad[i] << "    " << rho[i] << std::endl;
  }
  infile.close();

  // RESCALE
  for(int i=0;i<NARRAY;i++){
    rad[i] = rad[i]*rstar_initial;
    rho[i] = rho[i]*mstar_initial/pow(rstar_initial,3);
    p[i]   = p[i]*Ggrav*pow(mstar_initial,2)/pow(rstar_initial,4);
    menc_init[i] = menc_init[i]*mstar_initial;
  }

  
  
  // set the inner point mass based on excised mass
  Real menc_rin = Interpolate1DArrayEven(rad,menc_init, rmin, NARRAY );
  GM1 = Ggrav*menc_rin;
  Real GMenv = Ggrav*Interpolate1DArrayEven(rad,menc_init,1.01*rstar_initial, NARRAY) - GM1;


  // allocate the enclosed mass profile
  Real logr_min = log10(rmin);
  Real logr_max = log10(rmax);
  
  for(int i=0;i<NGRAV;i++){
    logr[i] = logr_min + (logr_max-logr_min)/(NGRAV-1)*i;
    menc[i] = Interpolate1DArrayEven(rad,menc_init, pow(10,logr[i]), NGRAV );
  }
  

    

  //ONLY enter ICs loop if this isn't a restart
  if(time==0){
    if(fixed_orbit){
      sma_fixed = pow((GM1+GM2+GMenv)/(Omega_orb_fixed*Omega_orb_fixed),1./3.);
      xi[0] = sma_fixed;
      xi[1] = 0.0;
      xi[2] = 0.0;  
      vi[0] = 0.0;
      vi[1]= Omega_orb_fixed*sma_fixed; 
      vi[2] = 0.0;

      Omega_envelope = fcorot*Omega_orb_fixed;
    }else{
      //Real vcirc = sqrt((GM1+GM2)/sma + accel*sma);    
      vcirc = sqrt((GM1+GM2+GMenv)/sma);
      Omega_orb = vcirc/sma;

      // set the initial conditions for the pos/vel of the secondary
      xi[0] = sma*(1.0 + ecc);  // apocenter
      xi[1] = 0.0;
      xi[2] = 0.0;
            
      vi[0] = 0.0;
      vi[1]= sqrt( vcirc*vcirc*(1.0 - ecc)/(1.0 + ecc) ); //v_apocenter
      vi[2] = 0.0;
    
      // now set the initial condition for Omega
      Omega[0] = 0.0;
      Omega[1] = 0.0;
      Omega[2] = 0.0;
    
      // In the case of a corotating frame,
      // subtract off the frame velocity and set Omega
      if(corotating_frame == 1){
	Omega[2] = Omega_orb;
	vi[1] -=  Omega[2]*xi[0]; 
      }
      
    
      // Angular velocity of the envelope (minus the frame?)
      Real f_pseudo = 1.0;
      if(ecc>0){
	f_pseudo=(1.0+7.5*pow(ecc,2) + 45./8.*pow(ecc,4) + 5./16.*pow(ecc,6));
	f_pseudo /= (1.0 + 3.0*pow(ecc,2) + 3./8.*pow(ecc,4))*pow(1-pow(ecc,2),1.5);
      }
      Omega_envelope = fcorot*Omega_orb*f_pseudo;

      // Decide whether to do pre-integration
      do_pre_integrate = (separation_start>sma*(1.0-ecc)) && (separation_start<sma*(1+ecc));
      
    } // end of if_not_fixed_orbit

    // save the ruser_mesh_data variables
    for(int i=0; i<3; i++){
      ruser_mesh_data[0](i)  = xi[i];
      ruser_mesh_data[1](i)  = vi[i];
      ruser_mesh_data[2](i)  = Omega[i];
      ruser_mesh_data[3](i)  = xcom[i];
      ruser_mesh_data[4](i)  = vcom[i];
    }
          
  }else{
    is_restart=1;
    trackfile_next_time=time;
  }
  
    
  // Print out some info
  if (Globals::my_rank==0){
    std::cout << "==========================================================\n";
    std::cout << "==========   SIMULATION INFO =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "time =" << time << "\n";
    std::cout << "Ggrav = "<< Ggrav <<"\n";
    std::cout << "gamma = "<< gamma_gas <<"\n";
    std::cout << "GM1 = "<< GM1 <<"\n";
    std::cout << "GM2 = "<< GM2 <<"\n";
    std::cout << "GMenv="<< GMenv << "\n";
    std::cout << "rstar_initial = "<< rstar_initial<<"\n";
    std::cout << "mstar_initial = "<< mstar_initial<<"\n";
    std::cout << "Omega_orb="<< Omega_orb << "\n";
    std::cout << "Omega_env="<< Omega_envelope << "\n";
    std::cout << "vphi_eq = "<< Omega_envelope*rstar_initial<<"\n";
    std::cout << "a = "<< sma <<"\n";
    std::cout << "e = "<< ecc <<"\n";
    std::cout << "P = "<< 6.2832*sqrt(sma*sma*sma/(GM1+GM2+GMenv)) << "\n";
    std::cout << "rsoft2 ="<<rsoft2<<"\n";
    std::cout << "corotating frame? = "<< corotating_frame<<"\n";
    std::cout << "gas backreaction? = "<< include_gas_backreaction<<"\n";
    std::cout << "particle substeping n="<<n_particle_substeps<<"\n";
    std::cout << "t_relax ="<<t_relax<<"\n";
    std::cout << "t_mass_on ="<<t_mass_on<<"\n";
    std::cout << "do_pre_integrate ="<<do_pre_integrate<<"\n";
    std::cout << "fixed_orbit ="<<fixed_orbit<<"\n";
    std::cout << "Omega_orb_fixed="<< Omega_orb_fixed << "\n";
    std::cout << "a_fixed = "<< sma_fixed <<"\n";
    std::cout << "==========================================================\n";
    std::cout << "==========   Particle        =============================\n";
    std::cout << "==========================================================\n";
    std::cout << "x ="<<xi[0]<<"\n";
    std::cout << "y ="<<xi[1]<<"\n";
    std::cout << "z ="<<xi[2]<<"\n";
    std::cout << "vx ="<<vi[0]<<"\n";
    std::cout << "vy ="<<vi[1]<<"\n";
    std::cout << "vz ="<<vi[2]<<"\n";
    std::cout << "==========================================================\n";
    std::cout << "cooling = " << cooling <<"\n";
    std::cout << "==========================================================\n";
  }
  
    


} // end





int RefinementCondition(MeshBlock *pmb)
{
  Real mindist=1.e99;
  Real rmin = 1.e99;
  int inregion = 0;
  for(int k=pmb->ks; k<=pmb->ke; k++){
    Real ph= pmb->pcoord->x3v(k);
    Real sin_ph = sin(ph);
    Real cos_ph = cos(ph);
    for(int j=pmb->js; j<=pmb->je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      Real cos_th = cos(th);
      for(int i=pmb->is; i<=pmb->ie; i++) {
	Real r = pmb->pcoord->x1v(i);
	Real x = r*sin_th*cos_ph;
	Real y = r*sin_th*sin_ph;
	Real z = r*cos_th;
	Real dist = std::sqrt(SQR(x-xi[0]) +
			      SQR(y-xi[1]) +
			      SQR(z-xi[2]) );
	mindist = std::min(mindist,dist);
	rmin    = std::min(rmin,r);
      }
    }
  }
  // derefine when away from pm & static region
  if( (mindist > 4.0*rsoft2) && rmin>x1_min_derefine  ) return -1;
  // refine near point mass 
  if(mindist <= 3.0*rsoft2) return 1;
   // otherwise do nothing
  return 0;
}



Real GetGM2factor(Real time){
  Real GM2_factor;

  // turn the gravity of the secondary on over time...
  if(time<t_relax+t_mass_on){
    if(time<t_relax){
      // if t<t_relax, do not apply the acceleration of the secondary to the gas
      GM2_factor = 0.0;
    }else{
      // turn on the gravity of the secondary over the course of t_mass_on after t_relax
      GM2_factor = (time-t_relax)/t_mass_on;
    }
  }else{
    // if we're outside of the relaxation times, turn the gravity of the secondary fully on
    GM2_factor = 1.0;
  }
  
  return GM2_factor;
}

void updateGM2(Real sep){
  GM2 = GM2i*std::min(sep/rstar_initial, 1.0); // linear decrease with separation
}



/// Cooling functions
Real kappa(Real rho, Real T)
{
  // From G. Knapp A403 Princeton Course Notes
  //Real Kes = 0.2*(1.0+X);
  //Real Ke = 0.2*(1.0+X)/((1.0+2.7e11*rho/(T*T))*(1.0+ pow((T/4.5e8),0.86) ));
  //Real Kk = 4.e25*(1+X)*(Z+1.e-3)*rho*pow(T,-3.5);
  //Real Khm = 1.1e-25*sqrt(Z) *sqrt(rho) * pow(T,7.7);
  //Real Km = 0.1*Z;
  // Real Krad = Km + 1.0/(1.0/Khm + 1.0/(Ke+Kk) );
  //return Krad;
  return mykappa;
}







// Source Function for two point masses
void TwoPointMass(MeshBlock *pmb, const Real time, const Real dt,  const AthenaArray<Real> *flux,
		  const AthenaArray<Real> &prim,
		  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
		  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar) 
{ 

  if(is_restart>0){
    // else this is a restart, read the current particle state
    for(int i=0; i<3; i++){
      xi[i]    = pmb->pmy_mesh->ruser_mesh_data[0](i);
      vi[i]    = pmb->pmy_mesh->ruser_mesh_data[1](i);
      Omega[i] = pmb->pmy_mesh->ruser_mesh_data[2](i);
      xcom[i]  = pmb->pmy_mesh->ruser_mesh_data[3](i);
      vcom[i]  = pmb->pmy_mesh->ruser_mesh_data[4](i);
    }
    // print some info
    if (Globals::my_rank==0){
      std::cout << "*** Setting initial conditions for t>0 ***\n";
      std::cout <<"xi="<<xi[0]<<" "<<xi[1]<<" "<<xi[2]<<"\n";
      std::cout <<"vi="<<vi[0]<<" "<<vi[1]<<" "<<vi[2]<<"\n";
      std::cout <<"Omega="<<Omega[0]<<" "<<Omega[1]<<" "<<Omega[2]<<"\n";
      std::cout <<"xcom="<<xcom[0]<<" "<<xcom[1]<<" "<<xcom[2]<<"\n";
      std::cout <<"vcom="<<vcom[0]<<" "<<vcom[1]<<" "<<vcom[2]<<"\n";
    }
    is_restart=0;
  }
  
  Real GM2_factor = GetGM2factor(time);

  // Gravitational acceleration from orbital motion
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    Real ph= pmb->pcoord->x3v(k);
    Real sin_ph = sin(ph);
    Real cos_ph = cos(ph);
    for (int j=pmb->js; j<=pmb->je; j++) {
      Real th= pmb->pcoord->x2v(j);
      Real sin_th = sin(th);
      Real cos_th = cos(th);
      for (int i=pmb->is; i<=pmb->ie; i++) {
	Real r = pmb->pcoord->x1v(i);
	
	// current position of the secondary
	Real x_2 = xi[0];
	Real y_2 = xi[1];
	Real z_2 = xi[2];
	Real d12c = pow(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2], 1.5);
	
	// spherical polar coordinates, get local cartesian           
	Real x = r*sin_th*cos_ph;
	Real y = r*sin_th*sin_ph;
	Real z = r*cos_th;
  
	Real d2  = sqrt(pow(x-x_2, 2) +
			pow(y-y_2, 2) +
			pow(z-z_2, 2) );
  
	//
	//  COMPUTE ACCELERATIONS 
	//
	// PM1
	//Real a_r1 = -GM1/pow(r,2);
	// cell volume avg'd version, see pointmass.cpp sourceterm code. 
	//Real a_r1 = -GM1*pmb->pcoord->coord_src1_i_(i)/r;
	Real GMenc1 = Ggrav*Interpolate1DArrayEven(logr,menc,log10(r) , NGRAV);
	Real a_r1 = -GMenc1*pmb->pcoord->coord_src1_i_(i)/r;
	//Real a_r1 = -GMenc1/pow(r,2);
	
	// PM2 gravitational accels in cartesian coordinates
	Real a_x = - GM2*GM2_factor * fspline(d2,rsoft2) * (x-x_2);   
	Real a_y = - GM2*GM2_factor * fspline(d2,rsoft2) * (y-y_2);  
	Real a_z = - GM2*GM2_factor * fspline(d2,rsoft2) * (z-z_2);
	
	// add the correction for the orbiting frame (relative to the COM)
	a_x += -  GM2*GM2_factor / d12c * x_2;
	a_y += -  GM2*GM2_factor / d12c * y_2;
	a_z += -  GM2*GM2_factor / d12c * z_2;
	
	if(corotating_frame == 1){
	  
	  Real vr  = prim(IVX,k,j,i);
	  Real vth = prim(IVY,k,j,i);
	  Real vph = prim(IVZ,k,j,i);
	  
	  // distance from the origin in cartesian (vector)
	  Real rxyz[3];
	  rxyz[0] = x;
	  rxyz[1] = y;
	  rxyz[2] = z;
	  
	  // get the cartesian velocities from the spherical (vector)
	  Real vgas[3];
	  vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	  vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	  vgas[2] = cos_th*vr - sin_th*vth;
	  
	  // add the centrifugal and coriolis terms
	  
	  // centrifugal
	  Real Omega_x_r[3], Omega_x_Omega_x_r[3];
	  cross(Omega,rxyz,Omega_x_r);
	  cross(Omega,Omega_x_r,Omega_x_Omega_x_r);
	  
	  a_x += - Omega_x_Omega_x_r[0];
	  a_y += - Omega_x_Omega_x_r[1];
	  a_z += - Omega_x_Omega_x_r[2];
	  
	  // coriolis
	  Real Omega_x_v[3];
	  cross(Omega,vgas,Omega_x_v);
	  
	  a_x += -2.0*Omega_x_v[0];
	  a_y += -2.0*Omega_x_v[1];
	  a_z += -2.0*Omega_x_v[2];
	}
	
	// add the gas acceleration of the frame of ref
	a_x += -agas1i[0];
	a_y += -agas1i[1];
	a_z += -agas1i[2];    
		
	// convert back to spherical
	Real a_r  = sin_th*cos_ph*a_x + sin_th*sin_ph*a_y + cos_th*a_z;
	Real a_th = cos_th*cos_ph*a_x + cos_th*sin_ph*a_y - sin_th*a_z;
	Real a_ph = -sin_ph*a_x + cos_ph*a_y;
	
	// add the PM1 accel
	a_r += a_r1;
	
	//
	// ADD SOURCE TERMS TO THE GAS MOMENTA/ENERGY
	//
	Real den = prim(IDN,k,j,i);
	
	Real src_1 = dt*den*a_r; 
	Real src_2 = dt*den*a_th;
	Real src_3 = dt*den*a_ph;

	// if the background is inert scale forces by envelope scalar
	if(inert_bg){
	  src_1 *= pmb->pscalars->r(0,k,j,i);
	  src_2 *= pmb->pscalars->r(0,k,j,i);
	  src_3 *= pmb->pscalars->r(0,k,j,i);
	}
	  
	
	// add the source term to the momenta  (source = - rho * a)
	cons(IM1,k,j,i) += src_1;
	cons(IM2,k,j,i) += src_2;
	cons(IM3,k,j,i) += src_3;
	
	// update the energy (source = - rho v dot a)
	//cons(IEN,k,j,i) += src_1*prim(IVX,k,j,i) + src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);
	cons(IEN,k,j,i) += src_1/den * 0.5*(flux[X1DIR](IDN,k,j,i) + flux[X1DIR](IDN,k,j,i+1));
	//cons(IEN,k,j,i) += src_2/den * 0.5*(flux[X2DIR](IDN,k,j,i) + flux[X2DIR](IDN,k,j+1,i)); //not sure why this seg-faults
	//cons(IEN,k,j,i) += src_3/den * 0.5*(flux[X3DIR](IDN,k,j,i) + flux[X3DIR](IDN,k+1,j,i));
	cons(IEN,k,j,i) += src_2*prim(IVY,k,j,i) + src_3*prim(IVZ,k,j,i);


	
	// APPLY LOCAL COOLING FUNCTION
	if(cooling){
	  Real denr0 = pmb->pscalars->r(0,k,j,i) * den;
	  //Real Hp = std::abs(prim(IPR,k,j,i)/( (prim(IPR,k,j,i+1)-prim(IPR,k,j,i-1))/(pmb->pcoord->x1v(i+1)-pmb->pcoord->x1v(i-1)) ));
	  //Hp = std::max(Hp,pmb->pcoord->x1v(i+1)-pmb->pcoord->x1v(i-1) );
	  
	  Real Sigma = std::max(denr0*rstar_initial,mH);
    
	  
	  Real mu = 0.61;
	  Real Temp = prim(IPR,k,j,i) * mu * mH / (den * kB);
	  Real Teq = fvir*GMenc1*mu*mH/(kB*r);  //pow( Lstar/(4*PI*sigmaSB*r*r), 0.25); // equilibrium temperature

	  Real kap = kappa(den,Temp);
	  Real tau = Sigma*kap;	  
	  
	  Real ueq = kB*Teq/(mu*mH*(gamma_gas-1));  // erg/g
	  Real u = prim(IPR,k,j,i)/(den*(gamma_gas-1)); // erg/g
	  	  
	  Real dudt = 4.0*sigmaSB*( pow(Teq,4) - pow(Temp,4))/(Sigma*tau + 1/kap);  //erg/g/s
	  Real t_therm = std::max( std::abs((ueq-u)/dudt) , 10.0*(pmb->pmy_mesh->dt) );
	  
	  Real exp_step = 1.0 - exp(-(pmb->pmy_mesh->dt) / t_therm);
	  
	  //std::cout<<"r="<<r<<"  tau="<<tau<<"  Temp="<<Temp<<"  Teq="<<Teq<<"  t_therm="<<t_therm<<"  exp="<<exp_step<<"\n";
	  
	  cons(IEN,k,j,i) +=  denr0*(ueq-u)*exp_step;

	    
	} // end coooling

	

      }
    }
  } // end loop over cells
  

}




//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical Coords HSE Envelope problem generator
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // local vars
  Real den, pres;

   // Prepare index bounds including ghost cells
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }
  
  // SETUP THE INITIAL CONDITIONS ON MESH
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {

	Real r  = pcoord->x1v(i);
	Real th = pcoord->x2v(j);
	Real ph = pcoord->x3v(k);

	
	Real sin_th = sin(th);
	Real cos_th = cos(th);
	Real Rcyl = r*sin_th;
	
	// get the density
	den = Interpolate1DArrayEven(rad,rho, r , NARRAY);
	den = std::max(den,da);
	
	// get the pressure 
	pres = Interpolate1DArrayEven(rad,p, r , NARRAY);
	pres = std::max(pres,pa);

	// set the density
	phydro->u(IDN,k,j,i) = den;
	
   	// set the momenta components
	phydro->u(IM1,k,j,i) = 0.0;
	phydro->u(IM2,k,j,i) = 0.0;

	
	if(r <= rstar_initial){
	  phydro->u(IM3,k,j,i) = den*(Omega_envelope*Rcyl - Omega[2]*Rcyl);
	}else{
	  phydro->u(IM3,k,j,i) = den*(Omega_envelope*pow(rstar_initial*sin_th,2)/Rcyl - Omega[2]*Rcyl);
	}

	//set the energy 
	phydro->u(IEN,k,j,i) = pres/(gamma_gas-1);
	phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
				     + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);

	// set the scalar
	if(r<rstar_initial){
	  pscalars->s(0,k,j,i) = 1.0*phydro->u(IDN,k,j,i);
	}else{
	  pscalars->s(0,k,j,i) = 1.e-30*phydro->u(IDN,k,j,i);
	}

	
	
      }
    }
  } // end loop over cells
  return;
} // end ProblemGenerator

//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//======================================================================================
void MeshBlock::UserWorkInLoop(void)
{
  Real time = pmy_mesh->time;
  Real dt = pmy_mesh->dt;
  Real tau;

  // if less than the relaxation time, apply 
  // a damping to the fluid velocities
  if(time < t_relax){
    tau = tau_relax_start;
    Real dex = log10(tau_relax_end)-log10(tau_relax_start);
    if(time > 0.2*t_relax){
      tau *= pow(10, dex*(time-0.2*t_relax)/(0.8*t_relax) );
    }
    if (Globals::my_rank==0){
      std::cout << "Relaxing: tau_damp ="<<tau<<std::endl;
    }
  } // time<t_relax

  for (int k=ks; k<=ke; k++) {
    Real ph= pcoord->x3v(k);
    //Real sin_ph = sin(ph);
    //Real cos_ph = cos(ph);
    for (int j=js; j<=je; j++) {
      Real th= pcoord->x2v(j);
      //Real sin_th = sin(th);
      //Real cos_th = cos(th);
      for (int i=is; i<=ie; i++) {
	Real r = pcoord->x1v(i);
	Real Rcyl = r*sin(th);
	Real den = phydro->u(IDN,k,j,i);
	Real GMenc1 = Ggrav*Interpolate1DArrayEven(logr,menc,log10(r) , NGRAV);
	pscalars->s(7,k,j,i) = GMenc1*pcoord->coord_src1_i_(i)*den; // neg epot     	

	if (time<t_relax){
	  Real vr  = phydro->u(IM1,k,j,i) / den;
	  Real vth = phydro->u(IM2,k,j,i) / den;
	  Real vph = phydro->u(IM3,k,j,i) / den;
	  Real vphEnv = Omega_envelope*std::min(Rcyl,1.5*rstar_initial) - Omega[2]*Rcyl;
	  Real vphBg = Omega_envelope*pow(rstar_initial*sin(th),2)/Rcyl - Omega[2]*Rcyl;
	  Real vphZone = vphBg + pscalars->r(0,k,j,i)*(vphEnv - vphBg);

	  Real a_damp_r =  - vr/tau;
	  Real a_damp_th = - vth/tau;
	  Real a_damp_ph = - (vph-vphZone)/tau;
	  //if(pscalars->r(0,k,j,i)>1.e-5){
	  //  a_damp_ph = - (vph-vphEnv)/tau * pscalars->r(0,k,j,i);
	  //}
	  phydro->u(IM1,k,j,i) += dt*den*a_damp_r;
	  phydro->u(IM2,k,j,i) += dt*den*a_damp_th;
	  phydro->u(IM3,k,j,i) += dt*den*a_damp_ph;
	  
	  phydro->u(IEN,k,j,i) += dt*den*a_damp_r*vr + dt*den*a_damp_th*vth + dt*den*a_damp_ph*vph; 
	
	
	  // set the lagrangian scalars (t<trelax)
	  pscalars->s(1,k,j,i) = r*den;
	  pscalars->s(2,k,j,i) = th*den;
	  pscalars->s(3,k,j,i) = ph*den;
	  
	  Real ek = 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i); 
	  pscalars->s(4,k,j,i) = phydro->u(IEN,k,j,i) - ek; //ei
	  pscalars->s(5,k,j,i) = ek; // ek
	  pscalars->s(6,k,j,i) = GMenc1*pcoord->coord_src1_i_(i)*den; // neg epot
	
	}//end time<t_relax
	
      }
    }
  } // end loop over cells                   
  

  return;
} // end of UserWorkInLoop


//========================================================================================
// MM
//! \fn void MeshBlock::MeshUserWorkInLoop(void)
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::MeshUserWorkInLoop(ParameterInput *pin){

  Real ai[3],acom[3];
  Real mg,mg_star;
  Mesh *pm = my_blocks(0)->pmy_mesh;


  // ONLY ON THE FIRST CALL TO THIS FUNCTION
  // (NOTE: DOESN'T WORK WITH RESTARTS)
  if((ncycle==0) && (fixed_orbit==false)){


    // initialize the COM position velocity
    SumComPosVel(pm, xi, vi, xcom, vcom, xcom_star, vcom_star, mg,mg_star);
        
    // kick the initial conditions back a half step (v^n-1/2)

    // first sum the gas accel if needed
    if(include_gas_backreaction == 1){
      SumGasOnParticleAccels(pm, xi,agas1i,agas2i);
    }

    ParticleAccels(xi,vi,ai);
    kick(-0.5*dt,xi,vi,ai);

    // Integrate from apocenter to separation_start
    if( do_pre_integrate ) {
      Real sep,vel,dt_pre_integrator;
      int n_steps_pre_integrator;

      SumComPosVel(pm, xi, vi, xcom, vcom, xcom_star, vcom_star, mg,mg_star);
      //Real GMenv = Ggrav*mg;
      Real GMenv = Ggrav*Interpolate1DArrayEven(rad,menc_init,1.01*rstar_initial, NARRAY) - GM1;

      
      for (int ii=1; ii<=1e8; ii++) {
	sep = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);
	vel = sqrt(vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2]);
	dt_pre_integrator = 1.e-4 * sep/vel;
	// check stopping condition
	n_steps_pre_integrator = ii;
	if (sep<separation_start) break;
	
	// add the particle acceleration to ai
	ParticleAccelsPreInt(GMenv,xi,vi,ai);
	// advance the particle
	particle_step(dt_pre_integrator,xi,vi,ai);
      }

      if (Globals::my_rank==0){
	std::cout << "Integrated to starting separation:"<<sep<<"\n";
	std::cout << " in "<<n_steps_pre_integrator<<" steps\n";
	if( std::abs(sep-separation_start) > 1.e-2*separation_start){
	  std::cout << "Pre-integrator failed!!! Exiting. \n";
	  SignalHandler::SetSignalFlag(SIGTERM); // make a clean exit
	}	
      }
    }
 
  } // ncycle=0, fixed_orbit = false

    
  // EVOLVE THE ORBITAL POSITION OF THE SECONDARY
  // do this on rank zero, then broadcast
  if (Globals::my_rank == 0 && time>t_relax){
    if(fixed_orbit){
      Real theta_orb = Omega_orb_fixed*time;
      xi[0] = sma_fixed*std::cos(theta_orb);
      xi[1] = sma_fixed*std::sin(theta_orb);
      xi[2] = 0.0;
      vi[0] = sma_fixed*Omega_orb_fixed*std::sin(theta_orb);
      vi[1] = sma_fixed*Omega_orb_fixed*std::cos(theta_orb);
      vi[2] = 0.0;

      SumComPosVel(pm, xi, vi, xcom, vcom, xcom_star, vcom_star, mg,mg_star);
    }else{
      for (int ii=1; ii<=n_particle_substeps; ii++) {
	// add the particle acceleration to ai
	ParticleAccels(xi,vi,ai);
	// advance the particle
	particle_step(dt/n_particle_substeps,xi,vi,ai);
      }
    }
  }
  
#ifdef MPI_PARALLEL
  // broadcast the position update from proc zero
  MPI_Bcast(xi,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vi,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif

  // update the ruser_mesh_data variables
  for(int i=0; i<3; i++){
    ruser_mesh_data[0](i)  = xi[i];
    ruser_mesh_data[1](i)  = vi[i];
    ruser_mesh_data[2](i)  = Omega[i];
    ruser_mesh_data[3](i)  = xcom[i];
    ruser_mesh_data[4](i)  = vcom[i];
  }

  // check the separation stopping conditions
  Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2] );
  if(d<dmin){
    dmin=d;
  }

  if (d<separation_stop_min){ 
    if (Globals::my_rank == 0) {
      std::cout << "### Stopping because binary separation d<separation_stop_min" << std::endl
		<< "d= " << d << " separation_stop_min="<<separation_stop_min<<std::endl;
    }
    SignalHandler::SetSignalFlag(SIGTERM); // make a clean exit
  }

  if (d>separation_stop_max){ 
    if (Globals::my_rank == 0) {
      std::cout << "### Stopping because binary separation d>separation_stop_max" << std::endl
		<< "d= " << d << " separation_stop_max="<<separation_stop_max<<std::endl;
    }
    SignalHandler::SetSignalFlag(SIGTERM); // make a clean exit
  }

  // check whether to trigger forced output
  if ((d<separation_stop_min) ||
      (d>separation_stop_max) ||
      (d<=output_next_sep) ){
    user_force_output = true;
    //output_next_sep -= dsep_output;
    if (Globals::my_rank == 0) {
      std::cout << "triggering user separation based output, d="<<d<<"\n";
    } 
    output_next_sep = floor(d/dsep_output)*dsep_output; // rounds to the nearest lower sep
  }
  
  
  // sum the enclosed mass profile for monopole gravity
  if(ncycle%update_grav_every == 0){
    SumMencProfile(pm,menc);
    if (Globals::my_rank == 0 ){
      std::cout << "enclosed mass updated... Menc(r=rstar_initial) = " << Interpolate1DArrayEven(logr,menc,log10(rstar_initial), NGRAV) <<"\n";
    }
  }

  // modify GM2 if needed
  if(update_gm2_sep){
    updateGM2(dmin);
  }
  
  // sum the gas->part accel for the next step
  if(include_gas_backreaction == 1 && time>t_relax){
    SumGasOnParticleAccels(pm, xi,agas1i,agas2i);
  }
  
  
  // write the output to the trackfile
  if(time >= trackfile_next_time || user_force_output ){
    SumComPosVel(pm, xi, vi, xcom, vcom, xcom_star, vcom_star, mg,mg_star);
    SumTrackfileDiagnostics(pm, xi, vi, lp, lg, ldo,
			    EK, EPot, EI, Edo, EK_star, EPot_star, EI_star, EK_ej, EPot_ej, EI_ej, M_star, mr1, mr12,mb,mu,
			    Eorb, Lz_star, Lz_orb,Lz_ej);
    WritePMTrackfile(pm,pin);
  }

  // std output
  if (Globals::my_rank == 0  & ncycle%100==0) {
    std::cout << "sep:  d="<<d<<"\n";
  } 
  
  
}


void WritePMTrackfile(Mesh *pm, ParameterInput *pin){
  
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign("pm_trackfile.dat");
    
    // open file for output
    FILE *pfile;
    std::stringstream msg;
    if((pfile = fopen(fname.c_str(),"a")) == NULL){
      msg << "### FATAL ERROR in function [WritePMTrackfile]" << std::endl
          << "Output file '" << fname << "' could not be opened";
      throw std::runtime_error(msg.str().c_str());
    }
  
    if(trackfile_number==0){
      fprintf(pfile,"#    ncycle                  ");
      fprintf(pfile,"time                ");
      fprintf(pfile,"dt                  ");
      fprintf(pfile,"m1                  ");
      fprintf(pfile,"m2                  ");
      fprintf(pfile,"x                   ");
      fprintf(pfile,"y                   ");
      fprintf(pfile,"z                   ");
      fprintf(pfile,"vx                  ");
      fprintf(pfile,"vy                  ");
      fprintf(pfile,"vz                  ");
      fprintf(pfile,"agas1x              ");
      fprintf(pfile,"agas1y              ");
      fprintf(pfile,"agas1z              ");
      fprintf(pfile,"agas2x              ");
      fprintf(pfile,"agas2y              ");
      fprintf(pfile,"agas2z              ");
      fprintf(pfile,"xcom                ");
      fprintf(pfile,"ycom                ");
      fprintf(pfile,"zcom                ");
      fprintf(pfile,"vxcom               ");
      fprintf(pfile,"vycom               ");
      fprintf(pfile,"vzcom               ");
      fprintf(pfile,"xcom_star           ");
      fprintf(pfile,"ycom_star           ");
      fprintf(pfile,"zcom_star           ");
      fprintf(pfile,"vxcom_star          ");
      fprintf(pfile,"vycom_star          ");
      fprintf(pfile,"vzcom_star          ");
      fprintf(pfile,"lpz                 ");
      fprintf(pfile,"lgz                 ");
      fprintf(pfile,"ldoz                ");
      fprintf(pfile,"EK                  ");
      fprintf(pfile,"EPot                ");
      fprintf(pfile,"EI                  ");
      fprintf(pfile,"Edo                 ");
      fprintf(pfile,"EK_star             ");
      fprintf(pfile,"EPot_star           ");
      fprintf(pfile,"EI_star             ");
      fprintf(pfile,"EK_ej               ");
      fprintf(pfile,"EPot_ej             ");
      fprintf(pfile,"EI_ej               ");
      fprintf(pfile,"M_star              ");
      fprintf(pfile,"Lz_star             ");
      fprintf(pfile,"mr1                 ");
      fprintf(pfile,"mr12                ");
      fprintf(pfile,"mb                  ");
      fprintf(pfile,"mu                  ");
      fprintf(pfile,"Eorb                ");
      fprintf(pfile,"Lz_orb              ");
      fprintf(pfile,"Lz_ej               ");
      fprintf(pfile,"\n");
    }


    // write the data line
    fprintf(pfile,"%20i",pm->ncycle);
    fprintf(pfile,"%20.6e",pm->time);
    fprintf(pfile,"%20.6e",pm->dt);
    fprintf(pfile,"%20.6e",GM1/Ggrav);
    fprintf(pfile,"%20.6e",GM2/Ggrav);
    fprintf(pfile,"%20.6e",xi[0]);
    fprintf(pfile,"%20.6e",xi[1]);
    fprintf(pfile,"%20.6e",xi[2]);
    fprintf(pfile,"%20.6e",vi[0]);
    fprintf(pfile,"%20.6e",vi[1]);
    fprintf(pfile,"%20.6e",vi[2]);
    fprintf(pfile,"%20.6e",agas1i[0]);
    fprintf(pfile,"%20.6e",agas1i[1]);
    fprintf(pfile,"%20.6e",agas1i[2]);
    fprintf(pfile,"%20.6e",agas2i[0]);
    fprintf(pfile,"%20.6e",agas2i[1]);
    fprintf(pfile,"%20.6e",agas2i[2]);
    fprintf(pfile,"%20.6e",xcom[0]);
    fprintf(pfile,"%20.6e",xcom[1]);
    fprintf(pfile,"%20.6e",xcom[2]);
    fprintf(pfile,"%20.6e",vcom[0]);
    fprintf(pfile,"%20.6e",vcom[1]);
    fprintf(pfile,"%20.6e",vcom[2]);
    fprintf(pfile,"%20.6e",xcom_star[0]);
    fprintf(pfile,"%20.6e",xcom_star[1]);
    fprintf(pfile,"%20.6e",xcom_star[2]);
    fprintf(pfile,"%20.6e",vcom_star[0]);
    fprintf(pfile,"%20.6e",vcom_star[1]);
    fprintf(pfile,"%20.6e",vcom_star[2]);
    fprintf(pfile,"%20.6e",lp[2]);
    fprintf(pfile,"%20.6e",lg[2]);
    fprintf(pfile,"%20.6e",ldo[2]);
    fprintf(pfile,"%20.6e",EK);
    fprintf(pfile,"%20.6e",EPot);
    fprintf(pfile,"%20.6e",EI);
    fprintf(pfile,"%20.6e",Edo);
    fprintf(pfile,"%20.6e",EK_star);
    fprintf(pfile,"%20.6e",EPot_star);
    fprintf(pfile,"%20.6e",EI_star);
    fprintf(pfile,"%20.6e",EK_ej);
    fprintf(pfile,"%20.6e",EPot_ej);
    fprintf(pfile,"%20.6e",EI_ej);
    fprintf(pfile,"%20.6e",M_star);
    fprintf(pfile,"%20.6e",Lz_star);
    fprintf(pfile,"%20.6e",mr1);
    fprintf(pfile,"%20.6e",mr12);
    fprintf(pfile,"%20.6e",mb);
    fprintf(pfile,"%20.6e",mu);
    fprintf(pfile,"%20.6e",Eorb);
    fprintf(pfile,"%20.6e",Lz_orb);
    fprintf(pfile,"%20.6e",Lz_ej);
    fprintf(pfile,"\n");

    // close the file
    fclose(pfile);  

  } // end rank==0

  // increment counters
  trackfile_number++;
  trackfile_next_time += trackfile_dt;


  
  return;
}



void particle_step(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  // Leapfrog algorithm (KDK)

  // kick a full step
  kick(dt,xi,vi,ai);

  // drift a full step
  drift(dt,xi,vi,ai);
  
}

// kick the velocities dt using the accelerations given in ai
void kick(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  for (int i = 0; i < 3; i++){
    vi[i]   += dt*ai[i];
  }
}

// drift the velocities dt using the velocities given in vi
void drift(Real dt,Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){
  for (int i = 0; i < 3; i++){
    xi[i]   += dt*vi[i];
  }
}

void ParticleAccels(Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){

  Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);

  // fill in the accelerations for the orbiting frame
  for (int i = 0; i < 3; i++){
    ai[i] = - GM1/pow(d,3) * xi[i] - GM2/pow(d,3) * xi[i];
  } 
  
  // IF WE'RE IN A ROTATING FRAME
  if(corotating_frame == 1){
    Real Omega_x_r[3],Omega_x_Omega_x_r[3], Omega_x_v[3];
     
    // compute cross products 
    cross(Omega,xi,Omega_x_r);
    cross(Omega,Omega_x_r,Omega_x_Omega_x_r);
    cross(Omega,vi,Omega_x_v);
    
// fill in the accelerations for the rotating frame
    for (int i = 0; i < 3; i++){
      ai[i] += -Omega_x_Omega_x_r[i];
      ai[i] += -2.0*Omega_x_v[i];
    }
  }

  // add the gas acceleration to ai
  if(include_gas_backreaction == 1){
    for (int i = 0; i < 3; i++){
      ai[i]   += -agas1i[i]+agas2i[i];
    }
  }

}


void ParticleAccelsPreInt(Real GMenv, Real (&xi)[3],Real (&vi)[3],Real (&ai)[3]){

  Real d = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);

  // fill in the accelerations for the orbiting frame
  for (int i = 0; i < 3; i++){
    ai[i] = - (GM1+GMenv)/pow(d,3) * xi[i] - GM2/pow(d,3) * xi[i];
  } 
  
  // IF WE'RE IN A ROTATING FRAME
  if(corotating_frame == 1){
    Real Omega_x_r[3],Omega_x_Omega_x_r[3], Omega_x_v[3];
 
    // compute cross products 
    cross(Omega,xi,Omega_x_r);
    cross(Omega,Omega_x_r,Omega_x_Omega_x_r);
    cross(Omega,vi,Omega_x_v);
  
    // fill in the accelerations for the rotating frame
    for (int i = 0; i < 3; i++){
      ai[i] += -Omega_x_Omega_x_r[i];
      ai[i] += -2.0*Omega_x_v[i];
    }
  }

}









void SumGasOnParticleAccels(Mesh *pm, Real (&xi)[3],Real (&ag1i)[3],Real (&ag2i)[3]){

  Real m1 =  GM1/Ggrav;
  // start by setting accelerations / positions to zero
  for (int ii = 0; ii < 3; ii++){
    ag1i[ii] = 0.0;
    ag2i[ii] = 0.0;
  }
  
  MeshBlock *pmb=pm->my_blocks(0);
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);

  // Loop over MeshBlocks
  for (int b=0; b<pm->nblocal; ++b) {
    pmb = pm->my_blocks(b);
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real ph= pmb->pcoord->x3v(k);
      Real sin_ph = sin(ph);
      Real cos_ph = cos(ph);
      for (int j=pmb->js; j<=pmb->je; ++j) {
	Real th= pmb->pcoord->x2v(j);
	Real sin_th = sin(th);
	Real cos_th = cos(th);
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  // cell mass dm
	  Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  	  
	  Real r = pmb->pcoord->x1v(i);

	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  Real d2 = sqrt(pow(x-xi[0], 2) +
			 pow(y-xi[1], 2) +
			 pow(z-xi[2], 2) );
  
	  Real d1c = pow(r,3);

	  // if we're on the innermost zone of the innermost block, assuming reflecting bc
	  if((pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::reflect) && (i==pmb->is)) {
	    // inner-r face area of cell i
	    Real dA = pmb->pcoord->GetFace1Area(k,j,i);
	  	 
	    // spherical velocities
	    Real vr =  phyd->u(IM1,k,j,i) / phyd->u(IDN,k,j,i);
	    Real vth =  phyd->u(IM2,k,j,i) / phyd->u(IDN,k,j,i);
	    Real vph =  phyd->u(IM3,k,j,i) / phyd->u(IDN,k,j,i);

	    // get the cartesian velocities from the spherical (vector)
	    Real vgas[3];
	    vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	    vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	    vgas[2] = cos_th*vr - sin_th*vth;

	    Real dAvec[3];
	    dAvec[0] = dA*(x/r);
	    dAvec[1] = dA*(y/r);
	    dAvec[2] = dA*(z/r);

	    // pressure terms (surface force/m1)
	    // note: not including the surface forces because the BC doesn't provide an appropriate counterterm
	    //ag1i[0] +=  -phyd->w(IPR,k,j,i)*dAvec[0]/m1;
	    //ag1i[1] +=  -phyd->w(IPR,k,j,i)*dAvec[1]/m1;
	    //ag1i[2] +=  -phyd->w(IPR,k,j,i)*dAvec[2]/m1;

	    // momentum flux terms
	    Real dAv = vgas[0]*dAvec[0] + vgas[1]*dAvec[1] + vgas[2]*dAvec[2];
	    ag1i[0] += phyd->u(IDN,k,j,i)*dAv*vgas[0]/m1;
	    ag1i[1] += phyd->u(IDN,k,j,i)*dAv*vgas[1]/m1;
	    ag1i[2] += phyd->u(IDN,k,j,i)*dAv*vgas[2]/m1;

	  }

	  
	  
	   // gravitational accels in cartesian coordinates
	  
	  ag1i[0] += Ggrav*dm/d1c * x;
	  ag1i[1] += Ggrav*dm/d1c * y;
	  ag1i[2] += Ggrav*dm/d1c * z;
	  
	  ag2i[0] += Ggrav*dm * fspline(d2,rsoft2) * (x-xi[0]);
	  ag2i[1] += Ggrav*dm * fspline(d2,rsoft2) * (y-xi[1]);
	  ag2i[2] += Ggrav*dm * fspline(d2,rsoft2) * (z-xi[2]);
	  
	}
      }
    }//end loop over cells
  }//end loop over meshblocks


#ifdef MPI_PARALLEL
  // sum over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, ag1i, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, ag2i, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(ag1i,ag1i,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(ag2i,ag2i,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  // and broadcast the result
  MPI_Bcast(ag1i,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(ag2i,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif
    
}


void SumComPosVel(Mesh *pm, Real (&xi)[3], Real (&vi)[3],
		  Real (&xcomSum)[3],Real (&vcomSum)[3],
		  Real (&xcom_star)[3],Real (&vcom_star)[3],
		  Real &mg, Real &mg_star){

   mg = 0.0;
   Real m1 = GM1/Ggrav;
   Real m2 = GM2/Ggrav;

   Real xgcom[3], vgcom[3], xgcom_star[3], vgcom_star[3];
  
  // start by setting accelerations / positions to zero
  for (int ii = 0; ii < 3; ii++){
    xgcom[ii] = 0.0;
    vgcom[ii] = 0.0;
    xgcom_star[ii] = 0.0;
    vgcom_star[ii] = 0.0;
  }
  
  MeshBlock *pmb=pm->my_blocks(0);
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);

  // Loop over MeshBlocks
  for (int b=0; b<pm->nblocal; ++b) {
    pmb = pm->my_blocks(b);
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  // cell mass dm
	  Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  	  
	  //coordinates
	  Real r = pmb->pcoord->x1v(i);
	  Real th= pmb->pcoord->x2v(j);
	  Real ph= pmb->pcoord->x3v(k);

	    //get some angles
	  Real sin_th = sin(th);
	  Real cos_th = cos(th);
	  Real sin_ph = sin(ph);
	  Real cos_ph = cos(ph);
	  
	 
	  // spherical velocities
	  Real vr =  phyd->u(IM1,k,j,i) / phyd->u(IDN,k,j,i);
	  Real vth =  phyd->u(IM2,k,j,i) / phyd->u(IDN,k,j,i);
	  Real vph =  phyd->u(IM3,k,j,i) / phyd->u(IDN,k,j,i);

	  // Correct for rotation of the frame? [TBDW]

	  
	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  // get the cartesian velocities from the spherical (vector)
	  Real vgas[3];
	  vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	  vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	  vgas[2] = cos_th*vr - sin_th*vth;

	  // do the summation
	  mg += dm;
	  
	  xgcom[0] += dm*x;
	  xgcom[1] += dm*y;
	  xgcom[2] += dm*z;

	  vgcom[0] += dm*vgas[0];
	  vgcom[1] += dm*vgas[1];
	  vgcom[2] += dm*vgas[2];

	  // do the summation (within the star)
	  if( instar(phyd->u(IDN,k,j,i), r )==true ){
	    mg_star += dm;
	    
	    xgcom_star[0] += dm*x;
	    xgcom_star[1] += dm*y;
	    xgcom_star[2] += dm*z;

	    vgcom_star[0] += dm*vgas[0];
	    vgcom_star[1] += dm*vgas[1];
	    vgcom_star[2] += dm*vgas[2];
	  }

	  
	}
      }
    }//end loop over cells
  }//end loop over meshblocks

#ifdef MPI_PARALLEL
  // sum over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &mg, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &mg_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, xgcom, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, vgcom, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, xgcom_star, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, vgcom_star, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&mg,&mg,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&mg_star,&mg_star,1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(xgcom,xgcom,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(vgcom,vgcom,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(xgcom_star,xgcom_star,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(vgcom_star,vgcom_star,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  MPI_Bcast(&mg,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(xgcom,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vgcom,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(xgcom_star,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vgcom_star,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif

  // normalize to the total mass
  for (int ii = 0; ii < 3; ii++){
    xgcom[ii] /= mg;
    vgcom[ii] /= mg;
    xgcom_star[ii] /= mg_star;
    vgcom_star[ii] /= mg_star;    
  }
  
  // FINISH CALC OF COM
  for (int ii = 0; ii < 3; ii++){
    xcomSum[ii] = (xi[ii]*m2 + xgcom[ii]*mg)/(m1+m2+mg);
    vcomSum[ii] = (vi[ii]*m2 + vgcom[ii]*mg)/(m1+m2+mg);
    xcom_star[ii] = xgcom_star[ii]*mg_star/(m1+mg_star);
    vcom_star[ii] = vgcom_star[ii]*mg_star/(m1+mg_star); 
  }

#ifdef MPI_PARALLEL
  MPI_Bcast(xcomSum,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vcomSum,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(xcom_star,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(vcom_star,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif
  
}



void SumTrackfileDiagnostics(Mesh *pm, Real (&xi)[3], Real (&vi)[3],
			     Real (&lp)[3],Real (&lg)[3],Real (&ldo)[3],
			     Real &EK, Real &EPot, Real &EI,Real &Edo,
			     Real &EK_star, Real &EPot_star, Real &EI_star,
			     Real &EK_ej, Real &EPot_ej, Real &EI_ej,
			     Real &M_star, Real &mr1, Real &mr12,
			     Real &mb, Real &mu,
			     Real &Eorb, Real &Lz_star, Real &Lz_orb, Real &Lz_ej){


  Real m1 = GM1/Ggrav;
  Real m2 = GM2/Ggrav;
  Real d12 = sqrt(xi[0]*xi[0] + xi[1]*xi[1] + xi[2]*xi[2]);

 
  // start by setting accelerations / positions to zero
  EK = 0.0;
  EPot = 0.0;
  EI = 0.0;
  Edo = 0.0;
  EK_star = 0.0;
  EPot_star = 0.0;
  EI_star = 0.0;
  EK_ej = 0.0;
  EPot_ej = 0.0;
  EI_ej = 0.0;
  M_star = 0.0;
  mr1 = 0.0;
  mr12 = 0.0;
  Lz_star = 0.0;
  Eorb = 0.0;
  mb = 0.0;
  mu = 0.0;
  Lz_orb = 0.0;
  Lz_ej = 0.0;

  Real EPotg2 = 0.0;

  for (int ii = 0; ii < 3; ii++){
    lg[ii]  = 0.0;
    lp[ii]  = 0.0;
    ldo[ii] = 0.0;
  }
  
  // loop over cells here
  MeshBlock *pmb=pm->my_blocks(0);
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);
  
  // Loop over MeshBlocks
  for (int b=0; b<pm->nblocal; ++b) {
    pmb = pm->my_blocks(b);
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real ph= pmb->pcoord->x3v(k);
      Real sin_ph = sin(ph);
      Real cos_ph = cos(ph);
      for (int j=pmb->js; j<=pmb->je; ++j) {
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	Real th= pmb->pcoord->x2v(j);
	Real sin_th = sin(th);
	Real cos_th = cos(th);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  Real r = pmb->pcoord->x1v(i);

	  // cell mass dm
	  Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  
	  // outer-r face area of cell i
	  Real dA = pmb->pcoord->GetFace1Area(k,j,i+1);
	  	 
	  // spherical velocities
	  Real vr =  phyd->u(IM1,k,j,i) / phyd->u(IDN,k,j,i);
	  Real vth =  phyd->u(IM2,k,j,i) / phyd->u(IDN,k,j,i);
	  Real vph =  phyd->u(IM3,k,j,i) / phyd->u(IDN,k,j,i);

	  // Correct for rotation of the frame? [TBDW]
	  
	  // spherical polar coordinates, get local cartesian           
	  Real x = r*sin_th*cos_ph;
	  Real y = r*sin_th*sin_ph;
	  Real z = r*cos_th;

	  // get the cartesian velocities from the spherical (vector)
	  Real vgas[3];
	  vgas[0] = sin_th*cos_ph*vr + cos_th*cos_ph*vth - sin_ph*vph;
	  vgas[1] = sin_th*sin_ph*vr + cos_th*sin_ph*vth + cos_ph*vph;
	  vgas[2] = cos_th*vr - sin_th*vth;

	  // do the summation
	  // position rel to COM
	  Real rg[3];
	  rg[0] = x - xcom[0];
	  rg[1] = y - xcom[1];
	  rg[2] = z - xcom[2];

	  // momentum rel to COM
	  Real pg[3];
	  pg[0] = dm*(vgas[0] - vcom[0]);
	  pg[1] = dm*(vgas[1] - vcom[1]);
	  pg[2] = dm*(vgas[2] - vcom[2]);

	  // rxp
	  Real rxp[3];
	  cross(rg,pg,rxp);
	  for (int ii = 0; ii < 3; ii++){
	    lg[ii] += rxp[ii];
	  }

	  // Z-component of angular momentum outsize star
	  if( instar(phyd->u(IDN,k,j,i), r )==false ){
	    Lz_ej += rxp[2]*pmb->pscalars->r(0,k,j,i);
	  }

	  // now the flux of angular momentum off of the outer boundary of the grid
	  if(pmb->pcoord->x1f(i+1)==pm->mesh_size.x1max){
	    Real md = phyd->u(IDN,k,j,i)*vr*dA;
	    Real pd[3];
	    pd[0] = md*(vgas[0] - vcom[0]);
	    pd[1] = md*(vgas[1] - vcom[1]);
	    pd[2] = md*(vgas[2] - vcom[2]);

	    Real rxpd[3];
	    cross(rg,pd,rxpd);
	    for (int ii = 0; ii < 3; ii++){
	      ldo[ii] += rxpd[ii];
	    }
	    // energy flux (KE + TE)
	    Edo += 0.5*md*(SQR(vgas[0] - vcom[0])
			   +SQR(vgas[1] - vcom[1])
			   +SQR(vgas[2] - vcom[2]));
	    Edo += md*pmb->phydro->w(IPR,k,j,i)/((gamma_gas-1.0)*pmb->phydro->u(IDN,k,j,i));
	      
	  } //endif


	  // enclosed mass within different conditions
	  if(instar(phyd->u(IDN,k,j,i),r)==true){
	    M_star += dm;
	  }
	  if(r<1.0*rstar_initial){
	    mr1 += dm;
	  }
	  if(r<1.2*rstar_initial){
	    mr12 += dm;
	  }

	  // energies
	  Real d2 = std::sqrt(SQR(x-xi[0]) +
			      SQR(y-xi[1]) +
			      SQR(z-xi[2]) );
	  Real GMenc1 = Ggrav*Interpolate1DArrayEven(logr,menc, log10(r), NGRAV);
	  Real h = gamma_gas * pmb->phydro->w(IPR,k,j,i)/((gamma_gas-1.0)*pmb->phydro->u(IDN,k,j,i));
	  Real epot = -GMenc1*pmb->pcoord->coord_src1_i_(i) - GM2*pspline(d2,rsoft2);
	  Real ek = 0.5*(SQR(vgas[0]-vcom[0]) +SQR(vgas[1]-vcom[1]) +SQR(vgas[2]-vcom[2]));
	  Real bern = h+ek+epot;
	  // bound/unbound mass outside of the star (weighted with scalar)
	  if ( instar(phyd->u(IDN,k,j,i),r)==false ) {
	    if (bern < 0.0){
	      mb += dm*pmb->pscalars->r(0,k,j,i);
	    }else{
	      mu += dm*pmb->pscalars->r(0,k,j,i);
	    }
	  }
	  
	  // intertial frame energies
	  EK += ek*dm;
	  EPot += epot*dm;
	  EPotg2 += - GM2*pspline(d2,rsoft2)*dm;
	  EI += vol(i)* (pmb->phydro->u(IEN,k,j,i) -
			 0.5*(SQR(pmb->phydro->u(IM1,k,j,i))+SQR(pmb->phydro->u(IM2,k,j,i))
			      + SQR(pmb->phydro->u(IM3,k,j,i)))/pmb->phydro->u(IDN,k,j,i) );

	  // stellar frame energy / angular momentum
	  if(instar(phyd->u(IDN,k,j,i),r)==true ){
	    EK_star += vol(i)*0.5*(SQR(pmb->phydro->u(IM1,k,j,i))+SQR(pmb->phydro->u(IM2,k,j,i))
				   + SQR(pmb->phydro->u(IM3,k,j,i)))/pmb->phydro->u(IDN,k,j,i);
	    EI_star += vol(i)* (pmb->phydro->u(IEN,k,j,i) -
				0.5*(SQR(pmb->phydro->u(IM1,k,j,i))+SQR(pmb->phydro->u(IM2,k,j,i))
				     + SQR(pmb->phydro->u(IM3,k,j,i)))/pmb->phydro->u(IDN,k,j,i) );
	    EPot_star += -GMenc1*pmb->pcoord->coord_src1_i_(i)*dm;

	    Lz_star += pmb->phydro->u(IM3,k,j,i)*vol(i)*r*sin_th;
	  }else{
	    // ejecta (intertial frame)
	    EK_ej += ek*dm*pmb->pscalars->r(0,k,j,i);
	    EPot_ej += epot*dm*pmb->pscalars->r(0,k,j,i);
	    EI_ej += pmb->pscalars->r(0,k,j,i)*vol(i)* (pmb->phydro->u(IEN,k,j,i) -
						     0.5*(SQR(pmb->phydro->u(IM1,k,j,i))+SQR(pmb->phydro->u(IM2,k,j,i))
							  + SQR(pmb->phydro->u(IM3,k,j,i)))/pmb->phydro->u(IDN,k,j,i) );
	    
	  }
	  
	  	    
	}
      }
    }//end loop over cells
  }//end loop over meshblocks
 
  
#ifdef MPI_PARALLEL
  // sum over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, lg, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, ldo, 3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EK, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EPot, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EI, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &Edo, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EK_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EPot_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EI_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EK_ej, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EPot_ej, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EI_ej, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &Lz_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &M_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &mr1, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &mr12, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &mb, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &mu, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &Lz_ej, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &EPotg2, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(lg,lg,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(ldo,ldo,3, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EK, &EK, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EPot, &EPot, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EI, &EI, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&Edo, &Edo, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EK_star, &EK_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EPot_star, &EPot_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EI_star, &EI_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EK_ej, &EK_ej, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EPot_ej, &EPot_ej, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EI_ej, &EI_ej, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&Lz_star, &Lz_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&M_star, &M_star, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&mr1, &mr1, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&mr12, &mr12, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&mb, &mb, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&mu, &mu, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&Lz_ej, &Lz_ej, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
    MPI_Reduce(&EPotg2, &EPotg2, 1, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  MPI_Bcast(lg,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(ldo,3,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EK,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EPot,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EI,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&Edo,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EK_star,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EPot_star,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EI_star,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EK_ej,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EPot_ej,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EI_ej,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&M_star,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&Lz_star,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&mr1,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&mr12,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&mb,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&mu,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&Lz_ej,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  MPI_Bcast(&EPotg2,1,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
  
#endif

  M_star += m1;
  mr1 += m1;
  mr12 += m1;
  EPot -= GM1*m2/d12;

  // calculate the particle angular momenta
  Real r1[3], r2[3], p1[3], p2[3], r1xp1[3], r2xp2[3];
  
  for (int ii = 0; ii < 3; ii++){
    r1[ii] = -xcom[ii];
    p1[ii] = -m1*vcom[ii];

    r2[ii] = xi[ii] - xcom[ii];
    p2[ii] = m2*(vi[ii] - vcom[ii]);
  }

  cross(r1,p1,r1xp1);
  cross(r2,p2,r2xp2);

  for (int ii = 0; ii < 3; ii++){
    lp[ii] = r1xp1[ii] + r2xp2[ii];
  }


  // calculate the orbital energy and angular momenta
  Real v1_sq = SQR(vcom_star[0]-vcom[0]) + SQR(vcom_star[1]-vcom[1]) + SQR(vcom_star[2]-vcom[2]);
  Real v2_sq = SQR(vi[0]-vcom[0]) + SQR(vi[1]-vcom[1]) + SQR(vi[2]-vcom[2]);
  EK += 0.5*m1*(SQR(vcom[0]) + SQR(vcom[1]) + SQR(vcom[2]));
  EK += 0.5*m2*v2_sq;
  Eorb = 0.5*M_star*v1_sq + 0.5*m2*v2_sq - GM1*m2/d12 + EPotg2;
  Real Lz_1 = M_star*((xcom_star[0]-xcom[0])*(vcom_star[1]-vcom[1])
		      -(xcom_star[1]-xcom[1])*(vcom_star[0]-vcom[0]));
  Lz_orb = Lz_1 + r2xp2[2];

  
}



void SumMencProfile(Mesh *pm, Real (&menc)[NGRAV]){

  Real m1 =  GM1/Ggrav;
  // start by setting enclosed mass at each radius to zero
  for (int ii = 0; ii <NGRAV; ii++){
    menc[ii] = 0.0;
  }
  
  MeshBlock *pmb=pm->my_blocks(0);
  AthenaArray<Real> vol;
  
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  vol.NewAthenaArray(ncells1);

  // Loop over MeshBlocks
  for (int b=0; b<pm->nblocal; ++b) {
    pmb = pm->my_blocks(b);
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
	pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
	for (int i=pmb->is; i<=pmb->ie; ++i) {
	  // cell mass dm
	  Real dm = vol(i) * phyd->u(IDN,k,j,i);
	  Real logr_cell = log10( pmb->pcoord->x1v(i) );

	  // loop over radii in profile
	  for (int ii = 0; ii <NGRAV; ii++){
	    if( logr_cell < logr[ii] ){
	      menc[ii] += dm;
	    }
	  }
	  
	}
      }
    }//end loop over cells
  }//end loop over meshblocks

#ifdef MPI_PARALLEL
  // sum over all ranks, add m1
  if (Globals::my_rank == 0) {
    for (int ii = 0; ii <NGRAV; ii++){
      menc[ii] += m1;
    }
    MPI_Reduce(MPI_IN_PLACE, menc, NGRAV, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(menc,menc,NGRAV, MPI_ATHENA_REAL, MPI_SUM, 0,MPI_COMM_WORLD);
  }

  
  // and broadcast the result
  MPI_Bcast(menc,NGRAV,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif
    
}



//--------------------------------------------------------------------------------------
//! \fn void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
		    FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // copy hydro variables into ghost zones, don't allow inflow
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVX)) {
      for (int k=ks; k<=ke; ++k) {
	for (int j=js; j<=je; ++j) {
#pragma simd
	  for (int i=1; i<=(NGHOST); ++i) {
	    prim(IVX,k,j,ie+i) =  std::max( 0.0, prim(IVX,k,j,(ie-i+1)) );  // positive velocities only
	  }
	}}
    } else {
      for (int k=ks; k<=ke; ++k) {
	for (int j=js; j<=je; ++j) {
#pragma simd
	  for (int i=1; i<=(NGHOST); ++i) {
	    prim(n,k,j,ie+i) = prim(n,k,j,(ie-i+1));
	  }
	}}
    }
  }


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
	}
      }}

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
	}
      }}

    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
	for (int i=1; i<=(NGHOST); ++i) {
	  b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
	}
      }}
  }

  return;
}






// 1D Interpolation that assumes EVEN spacing in x array

Real Interpolate1DArrayEven(Real *x,Real *y,Real x0, int length){ 
  // check the lower bound
  if(x[0] >= x0){
    //std::cout << "hit lower bound!\n";
    return y[0];
  }
  // check the upper bound
  if(x[length-1] <= x0){
    //std::cout << "hit upper bound!\n";
    return y[length-1];
  }

  int i = floor( (x0-x[0])/(x[1]-x[0]) );
  
  // if in the interior, do a linear interpolation
  if (x[i+1] >= x0){ 
    Real dx =  (x[i+1]-x[i]);
    Real d = (x0 - x[i]);
    Real s = (y[i+1]-y[i]) /dx;
    return s*d + y[i];
  }
  // should never get here, -9999.9 represents an error
  return -9999.9;
}



Real fspline(Real r, Real eps){
  // Hernquist & Katz 1989 spline kernel F=-GM r f(r,e) EQ A2
  Real u = r/eps;
  Real u2 = u*u;

  if (u<1.0){
    return pow(eps,-3) * (4./3. - 1.2*pow(u,2) + 0.5*pow(u,3) );
  } else if(u<2.0){
    return pow(r,-3) * (-1./15. + 8./3.*pow(u,3) - 3.*pow(u,4) + 1.2*pow(u,5) - 1./6.*pow(u,6));
  } else{
    return pow(r,-3);
  }

}


Real pspline(Real r, Real eps){
  Real u = r/eps;
  if (u<1.0){
    return -2/eps *(1./3.*pow(u,2) -0.15*pow(u,4) + 0.05*pow(u,5)) +7./(5.*eps);
  } else if(u<2.0){
    return -1./(15.*r) - 1/eps*( 4./3.*pow(u,2) - pow(u,3) + 0.3*pow(u,4) -1./30.*pow(u,5)) + 8./(5.*eps);
  } else{
    return 1/r;
  }

}







void cross(Real (&A)[3],Real (&B)[3],Real (&AxB)[3]){
  // set the vector AxB = A x B
  AxB[0] = A[1]*B[2] - A[2]*B[1];
  AxB[1] = A[2]*B[0] - A[0]*B[2];
  AxB[2] = A[0]*B[1] - A[1]*B[0];
}


bool instar(Real den, Real r){
  return ((den>1.e-3*mstar_initial/pow(rstar_initial,3) ) & (r<2*rstar_initial));
}
