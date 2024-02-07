/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(d3, PairD3)

#else

#ifndef LMP_PAIR_D3
#define LMP_PAIR_D3
#define _USE_MATH_DEFINES

#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unordered_map>
#include <omp.h>
#include "pair.h"
#include "memory.h"
#include "atom.h"
#include "utils.h"
#include "error.h"
#include "comm.h"
#include "potential_file_reader.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "domain.h"
#include "math_extra.h"

namespace LAMMPS_NS {

    class PairD3 : public Pair {
    public:
        // Constructor
        PairD3(class LAMMPS*);
        // Destructor
        ~PairD3() override;

        void compute(int, int) override;
        void settings(int, char**) override;
        void coeff(int, char**) override;
        double init_one(int i, int j) override;
        void init_style() override;

        void write_restart(FILE*) override;
        void read_restart(FILE*) override;
        void write_restart_settings(FILE*) override;
        void read_restart_settings(FILE*) override;

    protected:
        virtual void allocate();

        int find_atomic_number(std::string&);
        int is_int_in_array(int*, int, int);
        void read_r0ab(class LAMMPS*, char*, int*, int);
        void get_limit_in_pars_array(int&, int&, int&, int&);
        void read_c6ab(class LAMMPS*, char*, int*, int);
        void setfuncpar(char*);

        void get_dC6_dCNij();
        void set_lattice_vectors();

        double get_rep_cell(double*, double*, double);
        void set_criteria(double, int*);

        void get_tau(double, double, double, double*);
        double get_distance_with_tau(double*, int, int);
        void gather_cn();

        int lin(int, int);
        void inv_cell(double**, double**);
        double get_cell_volume(double**);
        void reallocate_arrays();
        void shift_atom_coord(double*, double*);

        void initialize_array();
        void precalculate_tau_array();
        void calculate_force_coefficients();
        void get_forces();
        void update(int, int);
        void allocate_for_omp();
        void initialize_for_omp();

        /*--------- Constants ---------*/
        int maxat;              // maximum number of total atoms
        int max_elem;           // maximum of the element number
        int maxc;               // maximum coordination number references per element

        double au_to_ang;       // conversion factors (atomic unit --> angstrom)
        double au_to_kcal;      // conversion factors (atomic unit --> kcal)
        double au_to_ev;        // conversion factors (atomic unit --> eV)
        double c6conv;          //  J/mol nm^6 - > au

        double k1;              // global ad hoc parameters
        double k2;              // global ad hoc parameters
        double k3;              // global ad hoc parameters
        /*--------- Constants ---------*/

        /*--------- Parameters to read ---------*/
        double* r2r4 = nullptr;            // scale r4/r2 values of the atoms by sqrt(Z)
        double* rcov = nullptr;            // covalent radii
        int* mxc = nullptr;             // How large the grid for c6 interpolation
        double** r0ab = nullptr;           // cut-off radii for all element pairs
        double***** c6ab = nullptr;            // C6 for all element pairs
                                   //
        double s6, s18, rs6, rs8, rs18, alp, alp6, alp8;
        double rthr;              // R^2 distance to cutoff for C calculation
        double cn_thr;            // R^2 distance to cutoff for CN_calculation
        /*--------- Parameters to read ---------*/

        double** lat;                  // For conversion of coordination
        double** lat_inv = nullptr;              // For conversion of coordination
        double* lat_v_1 = nullptr;               // lattice coordination vector
        double* lat_v_2 = nullptr;               // lattice coordination vector
        double* lat_v_3 = nullptr;               // lattice coordination vector
        double* lat_cp_12 = nullptr;           // Cross product of lat_v_1, lat_v_2
        double* lat_cp_23 = nullptr;           // Cross product of lat_v_2, lat_v_3
        double* lat_cp_31 = nullptr;           // Cross product of lat_v_3, lat_v_1
        int* rep_vdw = nullptr;            // repetition of cell for calculating D3
        int* rep_cn = nullptr;             // repetition of cell for calculating
                                 // coordination number
        double* cn = nullptr;              // coordination numbers of the atoms
        double** x = nullptr;              // Global position of atoms
                                 // needed, as D3 calculation done in au

        int* iz = nullptr;
        double* dc6i = nullptr;           // dC6i(iat) saves dE_dsp/dCN(iat)
        double** f = nullptr;             // Global force of atoms
        double**** drij = nullptr;

        int n_save;             // to check whether the number of atoms has changed
        double disp_total;
        double** sigma = nullptr;

        double**** tau_vdw = nullptr;
        double**** tau_cn = nullptr;

        int* tau_idx_vdw = nullptr;
        int* tau_idx_cn = nullptr;
        int tau_idx_vdw_total_size;
        int tau_idx_cn_total_size;

        double* c6_ij_tot = nullptr;
        double* dc6_iji_tot = nullptr;
        double* dc6_ijj_tot = nullptr;

        /* ------------ For OpenMP running ------------ */
        double* dc6i_private = nullptr;
        double* disp_private = nullptr;
        double* f_private = nullptr;
        double* sigma_private = nullptr;
        double* cn_private = nullptr;
        /* ------------ For OpenMP running ------------ */
    };
}    // namespace LAMMPS_NS

#endif
#endif

