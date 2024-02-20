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
#include "neigh_request.h"
#include "domain.h"
#include "math_extra.h"

namespace LAMMPS_NS {

    class PairD3 : public Pair {
        friend class NeighRequest;
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
        void init_list(int, class NeighList *) override;

    protected:
        virtual void allocate();

        /* ------- Read parameters ------- */
        int find_atomic_number(std::string&);
        int is_int_in_array(int*, int, int);
        void read_r0ab(class LAMMPS*, char*, int*, int);
        void get_limit_in_pars_array(int&, int&, int&, int&);
        void read_c6ab(class LAMMPS*, char*, int*, int);
        void setfuncpar(char*);
        /* ------- Read parameters ------- */



        /* ------- Initialize ------- */
        void initialize_array();
        void reallocate_arrays();
        /* ------- Initialize ------- */

        void gather_cn();
        void neigh_style_force_compute();
        void neigh_style_get_force();

        /*--------- Constants ---------*/
        static const int maxc = 5;
        static constexpr double au_to_ang = 0.52917726;
        static constexpr double ang_to_au = 1 / 0.52917726;
        static constexpr double au_to_ev = 27.21138505;
        static constexpr double ev_to_au = 1 / 27.21138505;
        static constexpr double k1 = 16.0;
        static constexpr double k2 = 4.0 / 3.0;
        static constexpr double k3 = -4.0;
        /*--------- Constants ---------*/


        /*--------- Parameters to read ---------*/
        double* r2r4 = nullptr;            // scale r4/r2 values of the atoms by sqrt(Z)
        double* rcov = nullptr;            // covalent radii
        int* mxc = nullptr;                // How large the grid for c6 interpolation
        double** r0ab = nullptr;           // cut-off radii for all element pairs
        double***** c6ab = nullptr;        // C6 for all element pairs
        double s6, s18, rs6, rs8, rs18, alp, alp6, alp8;  // parameters for D3
        double rthr;              // R^2 distance to cutoff for C calculation
        double cn_thr;            // R^2 distance to cutoff for CN_calculation
        double sq_vdw_ang;        // square of the vdw radius in angstrom
        double sq_cn_ang;         // square of the CN cutoff in angstrom
        /*--------- Parameters to read ---------*/


        /*--------- Per-atom values/arrays ---------*/
        double* cn = nullptr;               // Coordination numbers
        double* dc6i = nullptr;             // dC6i(iat) saves dE_dsp/dCN(iat)
        /*--------- Per-atom values/arrays ---------*/


        /*---------- Global values ---------*/
        int n_save;                         // to check whether the number of atoms has changed
        /*---------- Global values ---------*/

        // neighbor lists
        static const int NEIGH_CN_FULL_ID = 0;
        static const int NEIGH_CN_HALF_ID = 1;
        static const int NEIGH_VDW_HALF_ID = 2;
        class NeighList *cn_full;
        class NeighList *cn_half;
        class NeighList *vdw_half;
    };
}    // namespace LAMMPS_NS

#endif
#endif

