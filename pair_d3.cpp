/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Hyungmin An (andynn@snu.ac.kr)
------------------------------------------------------------------------- */


#include "pair_d3.h"

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   Constructor (Required)
------------------------------------------------------------------------- */

PairD3::PairD3(LAMMPS* lmp) : Pair(lmp) {
    single_enable = 0;      // potential is not pair-wise additive.
    restartinfo = 0;        // Many-body potentials are usually not
                            // written to binary restart files.
    one_coeff = 1;          // Many-body potnetials typically read all
                            // parameters from a file, so only one
                            // pair_coeff statement is needed.
    manybody_flag = 1;
}

/* ----------------------------------------------------------------------
   Destructor (Required)
------------------------------------------------------------------------- */

PairD3::~PairD3() {
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(r2r4);
        memory->destroy(rcov);
        memory->destroy(mxc);
        memory->destroy(r0ab);
        memory->destroy(c6ab);

        memory->destroy(lat_v_1);
        memory->destroy(lat_v_2);
        memory->destroy(lat_v_3);

        memory->destroy(rep_vdw);
        memory->destroy(rep_cn);
        memory->destroy(cn);
        memory->destroy(x);

        memory->destroy(dc6i);
        memory->destroy(f);

        memory->destroy(sigma);

        memory->destroy(tau_vdw);
        memory->destroy(tau_cn);
        memory->destroy(tau_idx_vdw);
        memory->destroy(tau_idx_cn);

        memory->destroy(dc6_iji_tot);
        memory->destroy(dc6_ijj_tot);
        memory->destroy(c6_ij_tot);

        memory->destroy(dc6i_private);
        memory->destroy(disp_private);
        memory->destroy(f_private);
        memory->destroy(sigma_private);
        memory->destroy(cn_private);
    }
}

/* ----------------------------------------------------------------------
   Allocate all arrays (Required)
------------------------------------------------------------------------- */

void PairD3::allocate() {
    allocated = 1;

    /* atom->ntypes : # of elements; element index starts from 1 */
    int np1 = atom->ntypes + 1;

    memory->create(setflag, np1, np1,                "pair:setflag");
    memory->create(cutsq,   np1, np1,                "pair:cutsq");
    memory->create(r2r4,    np1,                     "pair:r2r4");
    memory->create(rcov,    np1,                     "pair:rcov");
    memory->create(mxc,     np1,                     "pair:mxc");
    memory->create(r0ab,    np1, np1,                "pair:r0ab");
    memory->create(c6ab,    np1, np1, MAXC, MAXC, 3, "pair:c6");

    memory->create(lat_v_1, 3,      "pair:lat");
    memory->create(lat_v_2, 3,      "pair:lat");
    memory->create(lat_v_3, 3,      "pair:lat");
    memory->create(rep_vdw, 3,      "pair:rep_vdw");
    memory->create(rep_cn,  3,      "pair:rep_cn");
    memory->create(sigma,   3, 3,   "pair:sigma");

    int natoms = atom->natoms;
    n_save = natoms;

    memory->create(cn,   natoms,    "pair:cn");
    memory->create(x,    natoms, 3, "pair:x");
    memory->create(dc6i, natoms,    "pair:dc6i");
    memory->create(f,    natoms, 3, "pair:f");

    // Initialization (by function)
    set_lattice_vectors();

    // Initialization
    for (int i = 1; i < np1; i++) {
        for (int j = 1; j < np1; j++) {
            setflag[i][j] = 0;
        }
    }

    for (int idx1 = 0; idx1 < np1;  idx1++) {
        for (int idx2 = 0; idx2 < np1;  idx2++) {
            for (int idx3 = 0; idx3 < MAXC; idx3++) {
                for (int idx4 = 0; idx4 < MAXC; idx4++) {
                    for (int idx5 = 0; idx5 < 3;    idx5++) {
                        c6ab[idx1][idx2][idx3][idx4][idx5] = -1;
                    }
                }
            }
        }
    }

    int n_ij_combination = natoms * (natoms + 1) / 2;
    memory->create(dc6_iji_tot, n_ij_combination, "pair_dc6_iji_tot");
    memory->create(dc6_ijj_tot, n_ij_combination, "pair_dc6_ijj_tot");
    memory->create(c6_ij_tot,   n_ij_combination, "pair_c6_ij_tot");

    int vdw_range_x = 2 * rep_vdw[0] + 1;
    int vdw_range_y = 2 * rep_vdw[1] + 1;
    int vdw_range_z = 2 * rep_vdw[2] + 1;
    memory->create(tau_vdw, vdw_range_x, vdw_range_y, vdw_range_z, 3, "pair:tau_vdw");
    tau_idx_vdw_total_size = vdw_range_x * vdw_range_y * vdw_range_z * 3;
    memory->create(tau_idx_vdw, tau_idx_vdw_total_size, "pair:tau_idx_vdw");

    int cn_range_x  = 2 * rep_cn[0] + 1;
    int cn_range_y  = 2 * rep_cn[1] + 1;
    int cn_range_z  = 2 * rep_cn[2] + 1;
    memory->create(tau_cn,  cn_range_x,  cn_range_y,  cn_range_z,  3, "pair:tau_cn");
    tau_idx_cn_total_size = cn_range_x * cn_range_y * cn_range_z * 3;
    memory->create(tau_idx_cn, tau_idx_cn_total_size, "pair:tau_idx_cn");

}

/* ----------------------------------------------------------------------
   Settings: read from pair_style (Required)
             pair_style   d3  rthr cn_thr
------------------------------------------------------------------------- */

void PairD3::settings(int narg, char **arg) {
    if (narg != 2) { error->all(FLERR, "Pair_style d3 needs two arguments: rthr, cn_thr"); }
    rthr   = utils::numeric(FLERR, arg[0], false, lmp);
    cn_thr = utils::numeric(FLERR, arg[1], false, lmp);
}


/* ----------------------------------------------------------------------
   finds atomic number (used in PairD3::coeff)
------------------------------------------------------------------------- */

int PairD3::find_atomic_number(std::string& key) {
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);
    if (key.length() == 1) { key += " "; }
    key.resize(2);

    std::vector<std::string> element_table = {
        "h ","he",
        "li","be","b ","c ","n ","o ","f ","ne",
        "na","mg","al","si","p ","s ","cl","ar",
        "k ","ca","sc","ti","v ","cr","mn","fe","co","ni","cu",
        "zn","ga","ge","as","se","br","kr",
        "rb","sr","y ","zr","nb","mo","tc","ru","rh","pd","ag",
        "cd","in","sn","sb","te","i ","xe",
        "cs","ba","la","ce","pr","nd","pm","sm","eu","gd","tb","dy",
        "ho","er","tm","yb","lu","hf","ta","w ","re","os","ir","pt",
        "au","hg","tl","pb","bi","po","at","rn",
        "fr","ra","ac","th","pa","u ","np","pu"
    };

    int atomic_number;
    for (size_t i = 0; i < element_table.size(); ++i) {
        if (element_table[i] == key) {
            atomic_number = i + 1;
            return atomic_number;
        }
    }

    // if not the case
    return -1;
}

/* ----------------------------------------------------------------------
   Check whether an integer value in an integer array (used in PairD3::coeff)
------------------------------------------------------------------------- */

int PairD3::is_int_in_array(int arr[], int size, int value) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == value) { return i; } // returns the index
    }
    return -1;
}

/* ----------------------------------------------------------------------
   Read r0ab values from r0ab.csv (used in PairD3::coeff)
------------------------------------------------------------------------- */

void PairD3::read_r0ab(LAMMPS* lmp, char* path_r0ab, int* atomic_numbers, int ntypes) {

    int idx_atom_1 = -1, idx_atom_2 = -1;
    double value = 0;
    int nparams_per_line = 94;
    int row_idx = 1;
    char* line;

    PotentialFileReader r0ab_reader(lmp, path_r0ab, "d3");

    while ((line = r0ab_reader.next_line(nparams_per_line))) {
        idx_atom_1 = is_int_in_array(atomic_numbers, ntypes, row_idx);
        // Skip for the other rows
        if (idx_atom_1 < 0) { row_idx++; continue; }
        try {
            ValueTokenizer r0ab_values(line);

            for (int col_idx=1; col_idx <= nparams_per_line; col_idx++) {
                value = r0ab_values.next_double();
                idx_atom_2 = is_int_in_array(atomic_numbers, ntypes, col_idx);
                if (idx_atom_2 < 0) { continue; }
                r0ab[idx_atom_1+1][idx_atom_2+1] = value / AU_TO_ANG;
            } // loop over column

            row_idx++;
        } catch (TokenizerException& e) {
            error->one(FLERR, e.what());
        } // loop over rows
    }
}

/* ----------------------------------------------------------------------
   Get atom pair indices and grid indices (used in PairD3::read_c6ab)
------------------------------------------------------------------------- */

void PairD3::get_limit_in_pars_array(int& idx_atom_1, int& idx_atom_2, int& idx_i, int& idx_j) {
    idx_i = 1;
    idx_j = 1;
    int shift = 100;

    while (idx_atom_1 > shift) {
        idx_atom_1 -= shift;
        idx_i++;
    }

    while (idx_atom_2 > shift) {
        idx_atom_2 -= shift;
        idx_j++;
    }
}

/* ----------------------------------------------------------------------
   Read c6ab values from c6ab.csv (used in PairD3::coeff)
------------------------------------------------------------------------- */

void PairD3::read_c6ab(LAMMPS* lmp, char* path_c6ab, int* atomic_numbers, int ntypes) {

    for (int i = 1; i <= ntypes; i++) { mxc[i] = 0; }

    int atom_number_1 = 0, atom_number_2 = 0, grid_i = 0, grid_j = 0;
    int idx_atom_1 = -1, idx_atom_2 = -1;
    double ref_c6 = 0, ref_cn1 = 0, ref_cn2 = 0;
    char* line;
    int nparams_per_line = 5;

    PotentialFileReader c6ab_reader(lmp, path_c6ab, "d3");

    while ((line = c6ab_reader.next_line(nparams_per_line))) {
        try {
            ValueTokenizer c6ab_values(line);
            ref_c6 = c6ab_values.next_double();
            atom_number_1 = static_cast<int>(c6ab_values.next_double());
            atom_number_2 = static_cast<int>(c6ab_values.next_double());
            get_limit_in_pars_array(atom_number_1, atom_number_2, grid_i, grid_j);
            idx_atom_1 = is_int_in_array(atomic_numbers, ntypes, atom_number_1);
            if ( idx_atom_1 < 0 ) { continue; }
            idx_atom_2 = is_int_in_array(atomic_numbers, ntypes, atom_number_2);
            if ( idx_atom_2 < 0 ) { continue; }
            ref_cn1 = c6ab_values.next_double();
            ref_cn2 = c6ab_values.next_double();

            mxc[idx_atom_1 + 1] = std::max(mxc[idx_atom_1 + 1], grid_i);
            mxc[idx_atom_2 + 1] = std::max(mxc[idx_atom_2 + 1], grid_j);
            c6ab[idx_atom_1 + 1][idx_atom_2 + 1][grid_i - 1][grid_j - 1][0] = ref_c6;
            c6ab[idx_atom_1 + 1][idx_atom_2 + 1][grid_i - 1][grid_j - 1][1] = ref_cn1;
            c6ab[idx_atom_1 + 1][idx_atom_2 + 1][grid_i - 1][grid_j - 1][2] = ref_cn2;
            c6ab[idx_atom_2 + 1][idx_atom_1 + 1][grid_j - 1][grid_i - 1][0] = ref_c6;
            c6ab[idx_atom_2 + 1][idx_atom_1 + 1][grid_j - 1][grid_i - 1][1] = ref_cn2;
            c6ab[idx_atom_2 + 1][idx_atom_1 + 1][grid_j - 1][grid_i - 1][2] = ref_cn1;

        } catch (TokenizerException& e) {
            error->one(FLERR, e.what());
        } // loop over rows
    }

}

/* ----------------------------------------------------------------------
   Set functional parameters (used in PairD3::coeff)
------------------------------------------------------------------------- */

void PairD3::setfuncpar(char* functional_name) {
    // set parameters for the given functionals
    // DFT-D3
    s6 = 1.0;
    alp = 14.0;
    rs18 = 1.0;

    // default def2-QZVP (almost basis set limit)
    std::unordered_map<std::string, int> commandMap = {
    { "slater-dirac-exchange", 1}, { "b-lyp", 2 },    { "b-p", 3 },       { "b97-d", 4 },      { "revpbe", 5 },
    { "pbe", 6 },                  { "pbesol", 7 },   { "rpw86-pbe", 8 }, { "rpbe", 9 },       { "tpss", 10 },
    { "b3-lyp", 11 },              { "pbe0", 12 },    { "hse06", 13 },    { "revpbe38", 14 },  { "pw6b95", 15 },
    { "tpss0", 16 },               { "b2-plyp", 17 }, { "pwpb95", 18 },   { "b2gp-plyp", 19 }, { "ptpss", 20 },
    { "hf", 21 },                  { "mpwlyp", 22 },  { "bpbe", 23 },     { "bh-lyp", 24 },    { "tpssh", 25 },
    { "pwb6k", 26 },               { "b1b95", 27 },   { "bop", 28 },      { "o-lyp", 29 },     { "o-pbe", 30 },
    { "ssb", 31 },                 { "revssb", 32 },  { "otpss", 33 },    { "b3pw91", 34 },    { "revpbe0", 35 },
    { "pbe38", 36 },               { "mpw1b95", 37 }, { "mpwb1k", 38 },   { "bmk", 39 },       { "cam-b3lyp", 40 },
    { "lc-wpbe", 41 },             { "m05", 42 },     { "m052x", 43 },    { "m06l", 44 },      { "m06", 45 },
    { "m062x", 46 },               { "m06hf", 47 },   { "hcth120", 48 }
    };

    int commandCode = commandMap[functional_name];
    switch (commandCode) {
    case 1: rs6 = 0.999; s18 = -1.957; rs18 = 0.697; break;
    case 2: rs6 = 1.094; s18 = 1.682; break;
    case 3: rs6 = 1.139; s18 = 1.683; break;
    case 4: rs6 = 0.892; s18 = 0.909; break;
    case 5: rs6 = 0.923; s18 = 1.010; break;
    case 6: rs6 = 1.217; s18 = 0.722; break;
    case 7: rs6 = 1.345; s18 = 0.612; break;
    case 8: rs6 = 1.224; s18 = 0.901; break;
    case 9: rs6 = 0.872; s18 = 0.514; break;
    case 10: rs6 = 1.166; s18 = 1.105; break;
    case 11: rs6 = 1.261; s18 = 1.703; break;
    case 12: rs6 = 1.287; s18 = 0.928; break;
    case 13: rs6 = 1.129; s18 = 0.109; break;
    case 14: rs6 = 1.021; s18 = 0.862; break;
    case 15: rs6 = 1.532; s18 = 0.862; break;
    case 16: rs6 = 1.252; s18 = 1.242; break;
    case 17: rs6 = 1.427; s18 = 1.022; s6 = 0.64; break;
    case 18: rs6 = 1.557; s18 = 0.705; s6 = 0.82; break;
    case 19: rs6 = 1.586; s18 = 0.760; s6 = 0.56; break;
    case 20: rs6 = 1.541; s18 = 0.879; s6 = 0.75; break;
    case 21: rs6 = 1.158; s18 = 1.746; break;
    case 22: rs6 = 1.239; s18 = 1.098; break;
    case 23: rs6 = 1.087; s18 = 2.033; break;
    case 24: rs6 = 1.370; s18 = 1.442; break;
    case 25: rs6 = 1.223; s18 = 1.219; break;
    case 26: rs6 = 1.660; s18 = 0.550; break;
    case 27: rs6 = 1.613; s18 = 1.868; break;
    case 28: rs6 = 0.929; s18 = 1.975; break;
    case 29: rs6 = 0.806; s18 = 1.764; break;
    case 30: rs6 = 0.837; s18 = 2.055; break;
    case 31: rs6 = 1.215; s18 = 0.663; break;
    case 32: rs6 = 1.221; s18 = 0.560; break;
    case 33: rs6 = 1.128; s18 = 1.494; break;
    case 34: rs6 = 1.176; s18 = 1.775; break;
    case 35: rs6 = 0.949; s18 = 0.792; break;
    case 36: rs6 = 1.333; s18 = 0.998; break;
    case 37: rs6 = 1.605; s18 = 1.118; break;
    case 38: rs6 = 1.671; s18 = 1.061; break;
    case 39: rs6 = 1.931; s18 = 2.168; break;
    case 40: rs6 = 1.378; s18 = 1.217; break;
    case 41: rs6 = 1.355; s18 = 1.279; break;
    case 42: rs6 = 1.373; s18 = 0.595; break;
    case 43: rs6 = 1.417; s18 = 0.000; break;
    case 44: rs6 = 1.581; s18 = 0.000; break;
    case 45: rs6 = 1.325; s18 = 0.000; break;
    case 46: rs6 = 1.619; s18 = 0.000; break;
    case 47: rs6 = 1.446; s18 = 0.000; break;
    /* DFTB3(zeta = 4.0), old deprecated parameters; case ("dftb3"); rs6 = 1.235; s18 = 0.673; */
    case 48: rs6 = 1.221; s18 = 1.206; break;
    default:
        error->all(FLERR, "Functional name unknown");
        break;
    }

    rs8 = rs18;
    alp6 = alp;
    alp8 = alp + 2.0;

}

/* ----------------------------------------------------------------------
   Coeff: read from pair_coeff (Required)
          pair_coeff * * path_r0ab.csv path_c6ab.csv functional element1 element2 ...
------------------------------------------------------------------------- */

void PairD3::coeff(int narg, char **arg) {
    if (!allocated) allocate();
    if (narg < 3) { error->all(FLERR, "Pair_coeff * * needs: r0ab.csv c6ab.csv functional element1 element2 ..."); }

    std::string element;
    int ntypes = atom->ntypes;
    int* atomic_numbers = (int*)malloc(sizeof(int)*ntypes);
    for (int i = 0; i < ntypes; i++) {
        element = arg[i+5];
        atomic_numbers[i] = find_atomic_number(element);
    }

    int count = 0;
    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = 1; j <= atom->ntypes; j++) {
            setflag[i][j] = 1;
            count++;
        }
    }

    if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

    /*
    scale r4/r2 values of the atoms by sqrt(Z)
    sqrt is also globally close to optimum
    together with the factor 1/2 this yield reasonable
    c8 for he, ne and ar. for larger Z, C8 becomes too large
    which effectively mimics higher R^n terms neglected due
    to stability reasons

    r2r4 =sqrt(0.5*r2r4(i)*dfloat(i)**0.5 ) with i=elementnumber
    the large number of digits is just to keep the results consistent
    with older versions. They should not imply any higher accuracy than
    the old values
    */
    double r2r4_ref[94] = {
         2.00734898,  1.56637132,  5.01986934,  3.85379032,  3.64446594,
         3.10492822,  2.71175247,  2.59361680,  2.38825250,  2.21522516,
         6.58585536,  5.46295967,  5.65216669,  4.88284902,  4.29727576,
         4.04108902,  3.72932356,  3.44677275,  7.97762753,  7.07623947,
         6.60844053,  6.28791364,  6.07728703,  5.54643096,  5.80491167,
         5.58415602,  5.41374528,  5.28497229,  5.22592821,  5.09817141,
         6.12149689,  5.54083734,  5.06696878,  4.87005108,  4.59089647,
         4.31176304,  9.55461698,  8.67396077,  7.97210197,  7.43439917,
         6.58711862,  6.19536215,  6.01517290,  5.81623410,  5.65710424,
         5.52640661,  5.44263305,  5.58285373,  7.02081898,  6.46815523,
         5.98089120,  5.81686657,  5.53321815,  5.25477007, 11.02204549,
        10.15679528,  9.35167836,  9.06926079,  8.97241155,  8.90092807,
         8.85984840,  8.81736827,  8.79317710,  7.89969626,  8.80588454,
         8.42439218,  8.54289262,  8.47583370,  8.45090888,  8.47339339,
         7.83525634,  8.20702843,  7.70559063,  7.32755997,  7.03887381,
         6.68978720,  6.05450052,  5.88752022,  5.70661499,  5.78450695,
         7.79780729,  7.26443867,  6.78151984,  6.67883169,  6.39024318,
         6.09527958, 11.79156076, 11.10997644,  9.51377795,  8.67197068,
         8.77140725,  8.65402716,  8.53923501,  8.85024712
    }; // atomic <r^2>/<r^4> values

    /*
    covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197)
    values for metals decreased by 10 %
    !      data rcov/
    !     .  0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67
    !     ., 1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96, 1.76, 1.54
    !     ., 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09
    !     ., 1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39
    !     ., 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26
    !     ., 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57
    !     ., 1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53
    !     ., 1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32
    !     ., 1.30, 1.30, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58
    !     ., 1.52, 1.53, 1.54, 1.55 /

    these new data are scaled with k2=4./3.  and converted a_0 via
    autoang=0.52917726d0
    */

    double rcov_ref[94] = {
        0.80628308, 1.15903197, 3.02356173, 2.36845659, 1.94011865,
        1.88972601, 1.78894056, 1.58736983, 1.61256616, 1.68815527,
        3.52748848, 3.14954334, 2.84718717, 2.62041997, 2.77159820,
        2.57002732, 2.49443835, 2.41884923, 4.43455700, 3.88023730,
        3.35111422, 3.07395437, 3.04875805, 2.77159820, 2.69600923,
        2.62041997, 2.51963467, 2.49443835, 2.54483100, 2.74640188,
        2.82199085, 2.74640188, 2.89757982, 2.77159820, 2.87238349,
        2.94797246, 4.76210950, 4.20778980, 3.70386304, 3.50229216,
        3.32591790, 3.12434702, 2.89757982, 2.84718717, 2.84718717,
        2.72120556, 2.89757982, 3.09915070, 3.22513231, 3.17473967,
        3.17473967, 3.09915070, 3.32591790, 3.30072128, 5.26603625,
        4.43455700, 4.08180818, 3.70386304, 3.98102289, 3.95582657,
        3.93062995, 3.90543362, 3.80464833, 3.82984466, 3.80464833,
        3.77945201, 3.75425569, 3.75425569, 3.72905937, 3.85504098,
        3.67866672, 3.45189952, 3.30072128, 3.09915070, 2.97316878,
        2.92277614, 2.79679452, 2.82199085, 2.84718717, 3.32591790,
        3.27552496, 3.27552496, 3.42670319, 3.30072128, 3.47709584,
        3.57788113, 5.06446567, 4.56053862, 4.20778980, 3.98102289,
        3.82984466, 3.85504098, 3.88023730, 3.90543362
    }; // covalent radii

    for (int i = 0; i < ntypes; i++) {
        r2r4[i+1] = r2r4_ref[atomic_numbers[i]-1];
        rcov[i+1] = rcov_ref[atomic_numbers[i]-1];
    }

    // set r0ab
    read_r0ab(lmp, arg[2], atomic_numbers, ntypes);

    // read c6ab
    read_c6ab(lmp, arg[3], atomic_numbers, ntypes);

    // read functional parameters
    setfuncpar(arg[4]);

    free(atomic_numbers);

}

/* ----------------------------------------------------------------------
   Get derivative of C6 w.r.t. CN (used in PairD3::compute)

   C6 = C6(CN_A, CN_B) == W(CN_A, CN_B) / Z(CN_A, CN_B)

   This gives below from chain rule:
   d(C6)/dr = d(C6)/d(CN_A) * d(CN_A)/dr + d(C6)/d(CN_B) * d(CN_B)/dr

   So we can pre-calculate the d(C6)/d(CN_A), d(C6)/d(CN_B) part.

   d(C6)/d(CN_i) = (dW/d(CN_i) * Z - W * dZ/d(CN_i)) / (W * W)
        W : "denominator"
        Z : "numerator"
        dW/d(CN_i) : "d_denominator_i"
        dZ/d(CN_j) : "d_numerator_j"

    Z = Sum( L_ij(CN_A, CN_B) * C6_ref(CN_A_i, CN_B_j) ) over i, j
    W = Sum( L_ij(CN_A, CN_B) ) over i, j

   And the resulting derivative term is saved into
   "dc6_iji_tot", "dc6_ijj_tot" array,
   where we can find the value of d(C6)/d(CN_i)
   by knowing the index of "iat", and "jat". ("idx_linij")

   Also, c6 values will also be saved into "c6_ij_tot" array.

   Here, as we only interested in *pair* of atoms, assume "iat" >= "jat".
   Then "idx_linij" = "jat + (iat + 1) * iat / 2" have the order below.

     idx_linij | j = 0  j = 1  j = 2  j = 3    ...
---------------------------------------------
        i = 0  |     0
        i = 1  |     1      2
        i = 2  |     3      4      5
        i = 3  |     6      7      8      9
          ...  |    ...    ...    ...    ...   ...

------------------------------------------------------------------------- */

void PairD3::get_dC6_dCNij() {

    int n = atom->natoms;

    #pragma omp parallel
    {
        double c6mem;
        double r, r_save;
        double c6ref;
        double cn_refi, cn_refj;
        double expterm, term;
        double numerator, denominator;
        double d_numerator_i, d_denominator_i, d_numerator_j, d_denominator_j;

        int    mxci, mxcj;      // max number of reference point for iat_type, jat_type
        double cni, cnj;        // Coordination number (CN) of iat, jat
        int    idx_linij;       // index for (iat, jat) pair

        #pragma omp for schedule(auto)
        for (int iat = n - 1; iat >= 0; iat--) {
            for (int jat = iat; jat >= 0; jat--) {
                cni  = cn[iat];
                mxci = mxc[(atom->type)[iat]];

                cnj  = cn[jat];
                mxcj = mxc[(atom->type)[jat]];

                c6mem           = -1e99;
                r_save          = 9999.0;
                numerator       = 0.0;
                denominator     = 0.0;
                d_numerator_i   = 0.0;
                d_denominator_i = 0.0;
                d_numerator_j   = 0.0;
                d_denominator_j = 0.0;

                idx_linij = jat + (iat + 1) * iat / 2;

                for (int a = 0; a < mxci; a++) {
                    for (int b = 0; b < mxcj; b++) {
                        c6ref = c6ab[(atom->type)[iat]][(atom->type)[jat]][a][b][0];

                        if (c6ref > 0) {
                            cn_refi = c6ab[(atom->type)[iat]][(atom->type)[jat]][a][b][1];
                            cn_refj = c6ab[(atom->type)[iat]][(atom->type)[jat]][a][b][2];

                            r = (cn_refi - cni) * (cn_refi - cni) + (cn_refj - cnj) * (cn_refj - cnj);
                            if (r < r_save) {
                                r_save = r;
                                c6mem = c6ref;
                            }

                            // Corresponds to L_ij (in D3 paper)
                            expterm          = exp(K3 * r);
                            // numerator and denominator of C6 (Z and W in D3 paper)
                            numerator       += c6ref * expterm;
                            denominator     += expterm;

                            expterm         *= 2.0 * K3;

                            term             = expterm * (cni - cn_refi);
                            d_numerator_i   += c6ref * term;
                            d_denominator_i += term;

                            term             = expterm * (cnj - cn_refj);
                            d_numerator_j   += c6ref * term;
                            d_denominator_j += term;
                        }
                    } // b
                } // a

                if (denominator > 1e-99) {
                    c6_ij_tot[idx_linij]   = numerator / denominator;
                    dc6_iji_tot[idx_linij] = ((d_numerator_i * denominator) - (d_denominator_i * numerator)) / (denominator * denominator);
                    dc6_ijj_tot[idx_linij] = ((d_numerator_j * denominator) - (d_denominator_j * numerator)) / (denominator * denominator);
                }
                else {
                    c6_ij_tot[idx_linij]   = c6mem;
                    dc6_iji_tot[idx_linij] = 0.0;
                    dc6_ijj_tot[idx_linij] = 0.0;
                }
            } // jat
        } // iat
    } // omp parallel
}

/* ----------------------------------------------------------------------
   Get lattice vectors (used in PairD3::compute)

   1) Save lattice vectors into "lat_v_1", "lat_v_2", "lat_v_3"
   2) Calculate repetition criteria for vdw, cn
   3) precaluclate tau (xyz shift due to cell repetition)

------------------------------------------------------------------------- */

void PairD3::set_lattice_vectors() {

    double boxxlo = domain->boxlo[0];
    double boxxhi = domain->boxhi[0];
    double boxylo = domain->boxlo[1];
    double boxyhi = domain->boxhi[1];
    double boxzlo = domain->boxlo[2];
    double boxzhi = domain->boxhi[2];
    double xy = domain->xy;
    double xz = domain->xz;
    double yz = domain->yz;

    lat_v_1[0] = (boxxhi - boxxlo) / AU_TO_ANG;
    lat_v_1[1] =               0.0;
    lat_v_1[2] =               0.0;
    lat_v_2[0] =                xy / AU_TO_ANG;
    lat_v_2[1] = (boxyhi - boxylo) / AU_TO_ANG;
    lat_v_2[2] =               0.0;
    lat_v_3[0] =                xz / AU_TO_ANG;
    lat_v_3[1] =                yz / AU_TO_ANG;
    lat_v_3[2] = (boxzhi - boxzlo) / AU_TO_ANG;

    set_lattice_repetition_criteria(rthr, rep_vdw);
    set_lattice_repetition_criteria(cn_thr, rep_cn);
}

/* ----------------------------------------------------------------------
   Set repetition criteria (used in PairD3::compute)

   Needed as Periodic Boundary Condition should be considered.

   As the cell may *not* be orthorhombic,
   the dot product should be used between x/y/z direction and
   corresponding cross product vector.
------------------------------------------------------------------------- */

void PairD3::set_lattice_repetition_criteria(double r_threshold, int* rep_v) {
    double r_cutoff = sqrt(r_threshold);
    double lat_cp_12[3], lat_cp_23[3], lat_cp_31[3];
    double cos_value;

    MathExtra::cross3(lat_v_1, lat_v_2, lat_cp_12);
    MathExtra::cross3(lat_v_2, lat_v_3, lat_cp_23);
    MathExtra::cross3(lat_v_3, lat_v_1, lat_cp_31);

    cos_value = MathExtra::dot3(lat_cp_23, lat_v_1) / MathExtra::len3(lat_cp_23);
    rep_v[0] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;
    cos_value = MathExtra::dot3(lat_cp_31, lat_v_2) / MathExtra::len3(lat_cp_31);
    rep_v[1] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;
    cos_value = MathExtra::dot3(lat_cp_12, lat_v_3) / MathExtra::len3(lat_cp_12);
    rep_v[2] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;
}

/* ----------------------------------------------------------------------
   Calculate Coordination Number (used in PairD3::compute)
------------------------------------------------------------------------- */

void PairD3::get_coordination_number() {

    int nthreads = omp_get_max_threads();
    int n = atom->natoms;

    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();

        double r, r2;        // rAB in the paper
        double damp;     // fractional coordinate number
        int idx1, idx2, idx3;
        double rx, ry, rz;

        #pragma omp for schedule(auto)
        for (int iat = n - 1; iat >= 0; iat--) {
            for (int jat = iat - 1; jat >= 0; jat--) {
                for (int k = tau_idx_cn_total_size - 1; k >= 0; k -= 3) {
                    idx1 = tau_idx_cn[k-2];
                    idx2 = tau_idx_cn[k-1];
                    idx3 = tau_idx_cn[k];

                    rx = x[jat][0] - x[iat][0] + tau_cn[idx1][idx2][idx3][0];
                    ry = x[jat][1] - x[iat][1] + tau_cn[idx1][idx2][idx3][1];
                    rz = x[jat][2] - x[iat][2] + tau_cn[idx1][idx2][idx3][2];
                    r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 <= cn_thr) {
                        r = sqrt(r2);
                        damp = 1.0 / (1.0 + exp(-K1 * (((rcov[(atom->type)[iat]] + rcov[(atom->type)[jat]]) / r) - 1.0)));
                        cn_private[ithread * n + iat] += damp;
                    }
                }
            } // iat > jat

            for (int jat = n - 1; jat > iat; jat--) {
                for (int k = tau_idx_cn_total_size - 1; k >= 0; k -= 3) {
                    idx1 = tau_idx_cn[k-2];
                    idx2 = tau_idx_cn[k-1];
                    idx3 = tau_idx_cn[k];
                    rx = x[jat][0] - x[iat][0] + tau_cn[idx1][idx2][idx3][0];
                    ry = x[jat][1] - x[iat][1] + tau_cn[idx1][idx2][idx3][1];
                    rz = x[jat][2] - x[iat][2] + tau_cn[idx1][idx2][idx3][2];
                    r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 <= cn_thr) {
                        r = sqrt(r2);
                        damp = 1.0 / (1.0 + exp(-K1 * (((rcov[(atom->type)[iat]] + rcov[(atom->type)[jat]]) / r) - 1.0)));
                        cn_private[ithread * n + iat] += damp;
                    }
                }
            } // iat < jat

            for (int k = tau_idx_cn_total_size - 1; k >= 0; k -= 3) {
                // skip for the same atoms
                idx1 = tau_idx_cn[k-2];
                idx2 = tau_idx_cn[k-1];
                idx3 = tau_idx_cn[k];
                if (   idx1 != rep_cn[0]
                    || idx2 != rep_cn[1]
                    || idx3 != rep_cn[2]) {
                    rx = tau_cn[idx1][idx2][idx3][0];
                    ry = tau_cn[idx1][idx2][idx3][1];
                    rz = tau_cn[idx1][idx2][idx3][2];
                    r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 <= cn_thr) {
                        r = sqrt(r2);
                        damp = 1.0 / (1.0 + exp(-K1 * (((rcov[(atom->type)[iat]] + rcov[(atom->type)[iat]]) / r) - 1.0)));
                        cn_private[ithread * n + iat] += damp;
                    }
                }
            } // iat == jat
        } // iat
    } // omp parallel

    for (int rank = 0; rank < nthreads; rank++) {
        for (int iat = 0; iat < n; iat++) {
            if (rank == 0) { cn[iat] = cn_private[rank * n + iat]; }
            else { cn[iat] += cn_private[rank * n + iat]; }
        }
    } // summation over results from each OpenMP threads

    get_dC6_dCNij();
}


/* ----------------------------------------------------------------------
   reallcate memory if the number of atoms has changed (used in PairD3::compute)
------------------------------------------------------------------------- */

void PairD3::reallocate_arrays() {

    /* -------------- Destroy previous arrays -------------- */
    memory->destroy(cn);
    memory->destroy(x);
    memory->destroy(dc6i);
    memory->destroy(f);
    memory->destroy(tau_idx_vdw);
    memory->destroy(tau_idx_cn);

    memory->destroy(dc6_iji_tot);
    memory->destroy(dc6_ijj_tot);
    memory->destroy(c6_ij_tot);

    memory->destroy(dc6i_private);
    memory->destroy(disp_private);
    memory->destroy(f_private);
    memory->destroy(sigma_private);
    memory->destroy(cn_private);

    /* -------------- Destroy previous arrays -------------- */

    /* -------------- Create new arrays -------------- */
    int natoms = atom->natoms;
    n_save = natoms;

    memory->create(cn, natoms, "pair:cn");
    memory->create(x, natoms, 3, "pair:x");
    memory->create(dc6i, natoms, "pair:dc6i");
    memory->create(f, natoms, 3, "pair:f");

    set_lattice_vectors();

    int n_ij_combination = natoms * (natoms + 1) / 2;
    memory->create(dc6_iji_tot, n_ij_combination, "pair_dc6_iji_tot");
    memory->create(dc6_ijj_tot, n_ij_combination, "pair_dc6_ijj_tot");
    memory->create(c6_ij_tot,   n_ij_combination, "pair_c6_ij_tot");

    int vdw_range_x = 2 * rep_vdw[0] + 1;
    int vdw_range_y = 2 * rep_vdw[1] + 1;
    int vdw_range_z = 2 * rep_vdw[2] + 1;
    memory->create(tau_vdw, vdw_range_x, vdw_range_y, vdw_range_z, 3, "pair:tau_vdw");
    tau_idx_vdw_total_size = vdw_range_x * vdw_range_y * vdw_range_z * 3;
    memory->create(tau_idx_vdw, tau_idx_vdw_total_size, "pair:tau_idx_vdw");

    int cn_range_x  = 2 * rep_cn[0] + 1;
    int cn_range_y  = 2 * rep_cn[1] + 1;
    int cn_range_z  = 2 * rep_cn[2] + 1;
    memory->create(tau_cn,  cn_range_x,  cn_range_y,  cn_range_z,  3, "pair:tau_cn");
    tau_idx_cn_total_size = cn_range_x * cn_range_y * cn_range_z * 3;
    memory->create(tau_idx_cn, tau_idx_cn_total_size, "pair:tau_idx_cn");

    allocate_for_omp();

    /* -------------- Create new arrays -------------- */
}

/* ----------------------------------------------------------------------
  Initialize atomic positions & types (used in PairD3::compute)

   As the default xyz from lammps does not assure that atoms are within unit cell,
   this function shifts atoms into the unit cell.
------------------------------------------------------------------------- */

void PairD3::load_atom_info() {
    double lat[3][3];
    lat[0][0] = lat_v_1[0];
    lat[0][1] = lat_v_2[0];
    lat[0][2] = lat_v_3[0];
    lat[1][0] = lat_v_1[1];
    lat[1][1] = lat_v_2[1];
    lat[1][2] = lat_v_3[1];
    lat[2][0] = lat_v_1[2];
    lat[2][1] = lat_v_2[2];
    lat[2][2] = lat_v_3[2];

    double det = lat[0][0] * lat[1][1] * lat[2][2]
               + lat[0][1] * lat[1][2] * lat[2][0]
               + lat[0][2] * lat[1][0] * lat[2][1]
               - lat[0][2] * lat[1][1] * lat[2][0]
               - lat[0][1] * lat[1][0] * lat[2][2]
               - lat[0][0] * lat[1][2] * lat[2][1];

    double lat_inv[3][3];
    lat_inv[0][0] = (lat[1][1] * lat[2][2] - lat[1][2] * lat[2][1]) / det;
    lat_inv[1][0] = (lat[1][2] * lat[2][0] - lat[1][0] * lat[2][2]) / det;
    lat_inv[2][0] = (lat[1][0] * lat[2][1] - lat[1][1] * lat[2][0]) / det;
    lat_inv[0][1] = (lat[0][2] * lat[2][1] - lat[0][1] * lat[2][2]) / det;
    lat_inv[1][1] = (lat[0][0] * lat[2][2] - lat[0][2] * lat[2][0]) / det;
    lat_inv[2][1] = (lat[0][1] * lat[2][0] - lat[0][0] * lat[2][1]) / det;
    lat_inv[0][2] = (lat[0][1] * lat[1][2] - lat[0][2] * lat[1][1]) / det;
    lat_inv[1][2] = (lat[0][2] * lat[1][0] - lat[0][0] * lat[1][2]) / det;
    lat_inv[2][2] = (lat[0][0] * lat[1][1] - lat[0][1] * lat[1][0]) / det;

    double a[3] = { 0.0 };
    for (int iat = 0; iat < atom->natoms; iat++) {
        for (int i = 0; i < 3; i++) {
            a[i] = lat_inv[i][0] * (atom->x)[iat][0] + lat_inv[i][1] * (atom->x)[iat][1] + lat_inv[i][2] * (atom->x)[iat][2];
            if      (a[i] > 1) { while (a[i] > 1) { a[i]--; } }
            else if (a[i] < 0) { while (a[i] < 0) { a[i]++; } }
        }

        for (int i = 0; i < 3; i++) {
            x[iat][i] = (lat[i][0] * a[0] + lat[i][1] * a[1] + lat[i][2] * a[2]) / AU_TO_ANG;
        }
    }
}

/* ----------------------------------------------------------------------
   Precalculate tau array
------------------------------------------------------------------------- */

void PairD3::precalculate_tau_array() {
    int xlim = rep_vdw[0];
    int ylim = rep_vdw[1];
    int zlim = rep_vdw[2];

    int index = 0;
    for (int taux = -xlim; taux <= xlim; taux++) {
        for (int tauy = -ylim; tauy <= ylim; tauy++) {
            for (int tauz = -zlim; tauz <= zlim; tauz++) {
                tau_vdw[taux + xlim][tauy + ylim][tauz + zlim][0] = lat_v_1[0] * taux + lat_v_2[0] * tauy + lat_v_3[0] * tauz;
                tau_vdw[taux + xlim][tauy + ylim][tauz + zlim][1] = lat_v_1[1] * taux + lat_v_2[1] * tauy + lat_v_3[1] * tauz;
                tau_vdw[taux + xlim][tauy + ylim][tauz + zlim][2] = lat_v_1[2] * taux + lat_v_2[2] * tauy + lat_v_3[2] * tauz;
                tau_idx_vdw[index++] = taux + xlim;
                tau_idx_vdw[index++] = tauy + ylim;
                tau_idx_vdw[index++] = tauz + zlim;
            }
        }
    }

    xlim = rep_cn[0];
    ylim = rep_cn[1];
    zlim = rep_cn[2];

    index = 0;
    for (int taux = -xlim; taux <= xlim; taux++) {
        for (int tauy = -ylim; tauy <= ylim; tauy++) {
            for (int tauz = -zlim; tauz <= zlim; tauz++) {
                tau_cn[taux + xlim][tauy + ylim][tauz + zlim][0] = lat_v_1[0] * taux + lat_v_2[0] * tauy + lat_v_3[0] * tauz;
                tau_cn[taux + xlim][tauy + ylim][tauz + zlim][1] = lat_v_1[1] * taux + lat_v_2[1] * tauy + lat_v_3[1] * tauz;
                tau_cn[taux + xlim][tauy + ylim][tauz + zlim][2] = lat_v_1[2] * taux + lat_v_2[2] * tauy + lat_v_3[2] * tauz;
                tau_idx_cn[index++] = taux + xlim;
                tau_idx_cn[index++] = tauy + ylim;
                tau_idx_cn[index++] = tauz + zlim;
            }
        }
    }
}


/* ----------------------------------------------------------------------
   Get force coefficients
------------------------------------------------------------------------- */

void PairD3::get_forces_without_dC6() {
    int n = atom->natoms;

    for (int dim = 0; dim < n; dim++) { dc6i[dim] = 0.0; }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            f[i][j] = 0.0;
        }
    }

    double disp = 0.0;                  // stores energy (sanity check)
    int nthreads = omp_get_max_threads();

    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();

        double c6 = 0.0;                    // To save C6
        double dc6iji = 0.0, dc6ijj = 0.0;  // Derivative of C6 values
        double r0 = 0.0;                    // To save R0 : Covalent bond length
        double r42 = 0.0;                   // To save r2r4
        int idx_linij = 0;
        double r = 0.0, r2 = 0.0;           // Atomic distance and square of it
        double r_inv = 0.0, r2_inv = 0.0;   // inverse of r
        double r6_inv = 0.0, r7_inv = 0.0; // Power of r
        double t6 = 0.0, t8 = 0.0;          // dummy variable for calculation
        double damp6 = 0.0, damp8 = 0.0;    // dummy variable for calculation
        double s8 = s18;                    // D3 parameter for 8th-power term (just use s8 from beginning?)
        double dc6_rest = 0.0;
        double x1;
        double disp_sum = 0.0;
        double tmp_v = 0.0;
        int idx1, idx2, idx3;
        double rij[3];
        double vec[3];
        double sigma_local[3][3] = {{ 0.0 }};
        const double r2_rthr = rthr;

        #pragma omp for schedule(auto)
        for (int iat =   n - 1; iat >= 0; iat--) {
            // iat != jat
            for (int jat = iat - 1; jat >= 0; jat--) {
                r0 = r0ab[(atom->type)[iat]][(atom->type)[jat]];
                r42 = r2r4[(atom->type)[iat]] * r2r4[(atom->type)[jat]];
                idx_linij = jat + (iat + 1) * iat / 2;

                for (int k = tau_idx_vdw_total_size - 1; k >= 0; k -= 3) {
                    // cutoff radius check
                    idx1 = tau_idx_vdw[k-2];
                    idx2 = tau_idx_vdw[k-1];
                    idx3 = tau_idx_vdw[k];
                    rij[0] = x[jat][0] - x[iat][0] + tau_vdw[idx1][idx2][idx3][0];
                    rij[1] = x[jat][1] - x[iat][1] + tau_vdw[idx1][idx2][idx3][1];
                    rij[2] = x[jat][2] - x[iat][2] + tau_vdw[idx1][idx2][idx3][2];
                    r2 = MathExtra::lensq3(rij);
                    if (r2 > r2_rthr) { continue; }

                    r2_inv = 1.0 / r2;
                    r = sqrt(r2);
                    r_inv = 1.0 / r;

                    // Calculates damping functions
                    // alp6 = 14.0, alp8 = 16.0
                    tmp_v = (rs6 * r0) * r_inv;
                    t6 = tmp_v;
                    t6 *= t6;       // ^2
                    t6 *= tmp_v;    // ^3
                    t6 *= t6;       // ^6
                    t6 *= tmp_v;    // ^7
                    t6 *= t6;       // ^14
                    damp6 = 1.0 / (1.0 + 6.0 * t6);
                    t6 *= damp6;    // pre-calculation
                    t8 = (rs8 * r0) * r_inv;
                    t8 *= t8;       // ^2
                    t8 *= t8;       // ^4
                    t8 *= t8;       // ^8
                    t8 *= t8;       // ^16
                    damp8 = 1.0 / (1.0 + 6.0 * t8);
                    t8 *= damp8;    // pre-calculation

                    c6 = c6_ij_tot[idx_linij];
                    r6_inv = r2_inv * r2_inv * r2_inv;
                    r7_inv = r6_inv * r_inv;

                    /* // d(r ^ (-6)) / d(r_ij) */
                    x1 = 6.0 * c6 * r7_inv * (s6 * damp6 * (14.0 * t6 - 1.0) + s8 * r42 * r2_inv * damp8 * (48.0 * t8 - 4.0)) * r_inv;

                    vec[0] = x1 * rij[0];
                    vec[1] = x1 * rij[1];
                    vec[2] = x1 * rij[2];

                    f_private[ithread * n * 3 + iat * 3    ] -= vec[0];
                    f_private[ithread * n * 3 + iat * 3 + 1] -= vec[1];
                    f_private[ithread * n * 3 + iat * 3 + 2] -= vec[2];
                    f_private[ithread * n * 3 + jat * 3    ] += vec[0];
                    f_private[ithread * n * 3 + jat * 3 + 1] += vec[1];
                    f_private[ithread * n * 3 + jat * 3 + 2] += vec[2];

                    sigma_local[0][0] += vec[0] * rij[0];
                    sigma_local[0][1] += vec[0] * rij[1];
                    sigma_local[0][2] += vec[0] * rij[2];
                    sigma_local[1][0] += vec[1] * rij[0];
                    sigma_local[1][1] += vec[1] * rij[1];
                    sigma_local[1][2] += vec[1] * rij[2];
                    sigma_local[2][0] += vec[2] * rij[0];
                    sigma_local[2][1] += vec[2] * rij[1];
                    sigma_local[2][2] += vec[2] * rij[2];

                    // in dC6_rest all terms BUT C6 - term is saved for the kat - loop
                    dc6_rest = (s6 * damp6 + 3.0 * s8 * r42 * damp8 * r2_inv) * r6_inv;
                    disp_sum -= dc6_rest * c6;
                    dc6iji = dc6_iji_tot[idx_linij];
                    dc6ijj = dc6_ijj_tot[idx_linij];
                    dc6i_private[n * ithread + iat] += dc6_rest * dc6iji;
                    dc6i_private[n * ithread + jat] += dc6_rest * dc6ijj;
                } // k
            } // iat != jat

            // iat == jat
            for (int k = tau_idx_vdw_total_size - 1; k >= 0; k -= 3) {
                // cutoff radius check
                idx1 = tau_idx_vdw[k-2];
                idx2 = tau_idx_vdw[k-1];
                idx3 = tau_idx_vdw[k];
                if (idx1 == rep_vdw[0] && idx2 == rep_vdw[1] && idx3 == rep_vdw[2]) { continue; }
                rij[0] = tau_vdw[idx1][idx2][idx3][0];
                rij[1] = tau_vdw[idx1][idx2][idx3][1];
                rij[2] = tau_vdw[idx1][idx2][idx3][2];
                r2 = MathExtra::lensq3(rij);
                // cutoff radius check
                if (r2 > rthr) { continue; }

                r2_inv = 1.0 / r2;
                r = sqrt(r2);
                r_inv = 1.0 / r;
                r0 = r0ab[(atom->type)[iat]][(atom->type)[iat]];

                // Calculates damping functions
                tmp_v = (rs6 * r0) * r_inv;
                tmp_v *= tmp_v * tmp_v * tmp_v * tmp_v * tmp_v * tmp_v; // ^7
                t6 = tmp_v * tmp_v; // ^14
                damp6 = 1.0 / (1.0 + 6.0 * t6);
                tmp_v = (rs8 * r0) * r_inv;
                tmp_v = tmp_v * tmp_v; // ^2
                tmp_v = tmp_v * tmp_v; // ^4
                tmp_v = tmp_v * tmp_v; // ^8
                t8 = tmp_v * tmp_v; // ^16
                damp8 = 1.0 / (1.0 + 6.0 * t8);

                idx_linij = iat + (iat + 1) * iat / 2;
                c6 = c6_ij_tot[idx_linij];
                r42 = r2r4[(atom->type)[iat]] * r2r4[(atom->type)[iat]];
                r6_inv = r2_inv * r2_inv * r2_inv;
                r7_inv = r6_inv * r_inv;

                /* // d(r ^ (-6)) / d(r_ij) */
                x1 = 0.5 * 6.0 * c6 * r7_inv * (s6 * damp6 * (alp6 * t6 * damp6 - 1.0) + s8 * r42 * r2_inv * damp8 * (3.0 * alp8 * t8 * damp8 - 4.0)) * r_inv;

                vec[0] = x1 * rij[0];
                vec[1] = x1 * rij[1];
                vec[2] = x1 * rij[2];

                sigma_local[0][0] += vec[0] * rij[0];
                sigma_local[0][1] += vec[0] * rij[1];
                sigma_local[0][2] += vec[0] * rij[2];
                sigma_local[1][0] += vec[1] * rij[0];
                sigma_local[1][1] += vec[1] * rij[1];
                sigma_local[1][2] += vec[1] * rij[2];
                sigma_local[2][0] += vec[2] * rij[0];
                sigma_local[2][1] += vec[2] * rij[1];
                sigma_local[2][2] += vec[2] * rij[2];

                // in dC6_rest all terms BUT C6 - term is saved for the kat - loop
                dc6_rest = (s6 * damp6 + 3.0 * s8 * r42 * damp8 * r2_inv) * r6_inv * 0.5;

                disp_sum -= dc6_rest * c6;
                dc6iji = dc6_iji_tot[idx_linij];
                dc6ijj = dc6_ijj_tot[idx_linij];
                dc6i_private[n * ithread + iat] += dc6_rest * dc6iji;
                dc6i_private[n * ithread + iat] += dc6_rest * dc6ijj;
            } // iat == jat
        } // iat

        disp_private[ithread] = disp_sum;  // calculate E_disp for sanity check

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sigma_private[ithread * 9 + i * 3 + j] = sigma_local[i][j];
            }
        }

    } // pragma omp parallel

    for (int i = 0; i < nthreads; i++) {
        for (int iat = 0; iat < n; iat++) {
            dc6i[iat] += dc6i_private[n * i + iat];
        }

        disp += disp_private[i];
    }

    disp_total = disp;
}


/* ----------------------------------------------------------------------
   Get forces
------------------------------------------------------------------------- */

void PairD3::get_forces_with_dC6() {
    // After calculating all derivatives dE/dr_ij w.r.t. distances,
    // the grad w.r.t.the coordinates is calculated dE / dr_ij * dr_ij / dxyz_i

    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            sigma[ii][jj] = 0.0;
        }
    }

    int n = atom->natoms;

    int nthreads = omp_get_max_threads();
    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();

        double rcovij;                  // sum of covalent radius
        double expterm, dcnn;
        double x1;
        double vec[3] = { 0.0 };
        double rij[4] = { 0.0 };            // Displacement vector (to calculate r)
        double r = 0.0, r2 = 0.0;           // Atomic distance and square of it
        double sigma_local[3][3] = {{ 0.0 }};
        int idx1, idx2, idx3;

        #pragma omp for schedule(auto)
        for (int iat = n - 1; iat >= 0; iat--) {
            // iat != jat
            for (int jat = iat - 1; jat >= 0; jat--) {
                for (int k = tau_idx_cn_total_size - 1; k >= 0 ; k -= 3) {
                    idx1 = tau_idx_cn[k-2];
                    idx2 = tau_idx_cn[k-1];
                    idx3 = tau_idx_cn[k];
                    rij[0] = x[jat][0] - x[iat][0] + tau_cn[idx1][idx2][idx3][0];
                    rij[1] = x[jat][1] - x[iat][1] + tau_cn[idx1][idx2][idx3][1];
                    rij[2] = x[jat][2] - x[iat][2] + tau_cn[idx1][idx2][idx3][2];
                    r2 = MathExtra::lensq3(rij);
                    // Assume rthr > cn_thr --> only check for cn_thr
                    if (r2 >= cn_thr) { continue; }
                    r = sqrt(r2);
                    rcovij = rcov[(atom->type)[iat]] + rcov[(atom->type)[jat]];
                    expterm = exp(-K1 * (rcovij / r - 1.0));
                    dcnn = -K1 * rcovij * expterm / (r2 * (expterm + 1.0) * (expterm + 1.0));
                    x1 = dcnn * (dc6i[iat] + dc6i[jat]);

                    vec[0] = x1 * rij[0] / r;
                    vec[1] = x1 * rij[1] / r;
                    vec[2] = x1 * rij[2] / r;

                    f_private[ithread * n * 3 + iat * 3    ] -= vec[0];
                    f_private[ithread * n * 3 + iat * 3 + 1] -= vec[1];
                    f_private[ithread * n * 3 + iat * 3 + 2] -= vec[2];
                    f_private[ithread * n * 3 + jat * 3    ] += vec[0];
                    f_private[ithread * n * 3 + jat * 3 + 1] += vec[1];
                    f_private[ithread * n * 3 + jat * 3 + 2] += vec[2];

                    sigma_local[0][0] += vec[0] * rij[0];
                    sigma_local[0][1] += vec[0] * rij[1];
                    sigma_local[0][2] += vec[0] * rij[2];
                    sigma_local[1][0] += vec[1] * rij[0];
                    sigma_local[1][1] += vec[1] * rij[1];
                    sigma_local[1][2] += vec[1] * rij[2];
                    sigma_local[2][0] += vec[2] * rij[0];
                    sigma_local[2][1] += vec[2] * rij[1];
                    sigma_local[2][2] += vec[2] * rij[2];

                } // k
            } // iat != jat

            // iat == jat
            for (int k = tau_idx_cn_total_size - 1; k >= 0 ; k -= 3) {
                idx1 = tau_idx_cn[k-2];
                idx2 = tau_idx_cn[k-1];
                idx3 = tau_idx_cn[k];
                if (idx1 == rep_cn[0] && idx2 == rep_cn[1] && idx3 == rep_cn[2]) { continue; }

                rij[0] = tau_cn[idx1][idx2][idx3][0];
                rij[1] = tau_cn[idx1][idx2][idx3][1];
                rij[2] = tau_cn[idx1][idx2][idx3][2];
                r2 = MathExtra::lensq3(rij);
                // Assume rthr > cn_thr --> only check for cn_thr
                if (r2 >= cn_thr) { continue; }
                r = sqrt(r2);
                rcovij = rcov[(atom->type)[iat]] + rcov[(atom->type)[iat]];
                expterm = exp(-K1 * (rcovij / r - 1.0));
                dcnn = -K1 * rcovij * expterm / (r2 * (expterm + 1.0) * (expterm + 1.0));
                x1 = dcnn * dc6i[iat];

                vec[0] = x1 * rij[0] / r;
                vec[1] = x1 * rij[1] / r;
                vec[2] = x1 * rij[2] / r;

                sigma_local[0][0] += vec[0] * rij[0];
                sigma_local[0][1] += vec[0] * rij[1];
                sigma_local[0][2] += vec[0] * rij[2];
                sigma_local[1][0] += vec[1] * rij[0];
                sigma_local[1][1] += vec[1] * rij[1];
                sigma_local[1][2] += vec[1] * rij[2];
                sigma_local[2][0] += vec[2] * rij[0];
                sigma_local[2][1] += vec[2] * rij[1];
                sigma_local[2][2] += vec[2] * rij[2];
            } // k
        } // iat == jat

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sigma_private[ithread * 9 + i * 3 + j] += sigma_local[i][j];
            }
        }
    } // omp parallel

    for (int rank = 0; rank < nthreads; rank++) {
        for (int iat = 0; iat < n; iat++) {
            for (int j = 0; j < 3; j++) {
                f[iat][j] += f_private[rank * n * 3 + iat * 3 + j];
            }
        }
    }

    for (int rank = 0; rank < nthreads; rank++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sigma[i][j] += sigma_private[rank * 9 + i * 3 + j];
            }
        }
    }
}

/* ----------------------------------------------------------------------
   Update energy, force, and stress
------------------------------------------------------------------------- */

void PairD3::update(int eflag, int vflag) {
    int n = atom->natoms;
    // Energy update
    if (eflag) { eng_vdwl += disp_total * AU_TO_EV; }

    double** f_local = atom->f;       // Local force of atoms
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            f_local[i][j] += f[i][j] * AU_TO_EV / AU_TO_ANG;
        }
    }

    // Stress update
    if (vflag) {
        virial[0] += sigma[0][0] * AU_TO_EV;
        virial[1] += sigma[1][1] * AU_TO_EV;
        virial[2] += sigma[2][2] * AU_TO_EV;
        virial[3] += sigma[0][1] * AU_TO_EV;
        virial[4] += sigma[0][2] * AU_TO_EV;
        virial[5] += sigma[1][2] * AU_TO_EV;
    }
}

void PairD3::allocate_for_omp() {

    int natoms = atom->natoms;
    int nthreads = omp_get_max_threads();
    dc6i_private = memory->create(dc6i_private, nthreads * natoms, "pair:dc6i_private");
    disp_private = memory->create(disp_private, nthreads, "pair:disp_private");
    f_private = memory->create(f_private, nthreads * natoms * 3, "pair:f_private");
    sigma_private = memory->create(sigma_private, nthreads * 3 * 3, "pair:sigma_private");
    cn_private = memory->create(cn_private, nthreads * natoms, "pair:cn_private");

}

void PairD3::initialize_for_omp() {

    int natoms = atom->natoms;
    int nthreads = omp_get_max_threads();
    for (int i = 0; i < natoms * nthreads;     i++) { dc6i_private[i]  = 0.0; }
    for (int i = 0; i < nthreads;              i++) { disp_private[i]  = 0.0; }
    for (int i = 0; i < 3 * natoms * nthreads; i++) { f_private[i]     = 0.0; }
    for (int i = 0; i < 3 * 3 * nthreads;      i++) { sigma_private[i] = 0.0; }
    for (int i = 0; i < nthreads * natoms;     i++) { cn_private[i]    = 0.0; }

}

/* ----------------------------------------------------------------------
   Compute : energy, force, and stress (Required)
------------------------------------------------------------------------- */

void PairD3::compute(int eflag, int vflag) {
    if (eflag || vflag)          { ev_setup(eflag, vflag); }
    if (dc6i_private == nullptr) { allocate_for_omp(); }
    if (atom->natoms != n_save)  { reallocate_arrays(); }

    initialize_for_omp();
    set_lattice_vectors();
    precalculate_tau_array();
    load_atom_info();

    get_coordination_number();

    get_forces_without_dC6();
    get_forces_with_dC6();
    update(eflag, vflag);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairD3::init_one(int i, int j) {
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
    // No need to count local neighbor in D3
    /* return std::sqrt(rthr * std::pow(au_to_ang, 2)); */
    return 0.0;
}

/* ----------------------------------------------------------------------
   init specific to this pair style (Optional)
------------------------------------------------------------------------- */

void PairD3::init_style() {
    neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairD3::write_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairD3::read_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairD3::write_restart_settings(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairD3::read_restart_settings(FILE *fp) {}
