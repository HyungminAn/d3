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
#include "neighbor.h"

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
    /*
    maxc = 5;
    au_to_ang = 0.52917726;
    //au_to_kcal = 627.509541;
    au_to_ev = 27.21138505;
    k1 = 16.0;
    k2 = 4.0 / 3.0;
    k3 = -4.0;
    */
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
    }
}

/* ----------------------------------------------------------------------
   Allocate all arrays (Required)
------------------------------------------------------------------------- */

void PairD3::allocate() {
    allocated = 1;

    /* atom->ntypes : # of elements; element index starts from 1 */
    int np1 = atom->ntypes + 1;
    memory->create(setflag, np1, np1, "pair:setflag");
    memory->create(cutsq, np1, np1, "pair:cutsq");
    memory->create(r2r4, np1, "pair:r2r4");
    memory->create(rcov, np1, "pair:rcov");
    memory->create(mxc, np1, "pair:mxc");
    memory->create(r0ab, np1, np1, "pair:r0ab");
    memory->create(c6ab, np1, np1, maxc, maxc, 3, "pair:c6");

    int natoms = atom->natoms;
    n_save = natoms;

    memory->create(cn, natoms, "pair:cn");
    memory->create(dc6i, natoms, "pair:dc6i");

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
}

void PairD3::reallocate_arrays() {
    /* -------------- Destroy previous arrays -------------- */
    memory->destroy(cn);
    memory->destroy(dc6i);
    /* -------------- Destroy previous arrays -------------- */

    /* -------------- Create new arrays -------------- */
    int natoms = atom->natoms;
    n_save = natoms;
    int nthreads = omp_get_max_threads();

    memory->create(cn, natoms, "pair:cn");
    memory->create(dc6i, natoms, "pair:dc6i");
    /* -------------- Create new arrays -------------- */
}
/* ----------------------------------------------------------------------
   Settings: read from pair_style (Required)
             pair_style   d3  rthr cn_thr
------------------------------------------------------------------------- */

void PairD3::settings(int narg, char **arg) {
    if (narg != 2) {
        error->all(FLERR, "Pair_style d3 needs two arguments: rthr, cn_thr");
    }
    rthr = utils::numeric(FLERR, arg[0], false, lmp);
    cn_thr = utils::numeric(FLERR, arg[1], false, lmp);

    sq_vdw_ang = rthr * au_to_ang * au_to_ang;
    sq_cn_ang = cn_thr * au_to_ang * au_to_ang;
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

    for (size_t i = 0; i < element_table.size(); ++i) {
        if (element_table[i] == key) {
            int atomic_number = i + 1;
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

    int nparams_per_line = 94;
    int row_idx = 1;
    char* line;

    PotentialFileReader r0ab_reader(lmp, path_r0ab, "d3");

    while ((line = r0ab_reader.next_line(nparams_per_line))) {
        const int idx_atom_1 = is_int_in_array(atomic_numbers, ntypes, row_idx);
        // Skip for the other rows
        if (idx_atom_1 < 0) { row_idx++; continue; }
        try {
            ValueTokenizer r0ab_values(line);

            for (int col_idx=1; col_idx <= nparams_per_line; col_idx++) {
                const double value = r0ab_values.next_double();
                const int idx_atom_2 = is_int_in_array(atomic_numbers, ntypes, col_idx);
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

    int grid_i = 0, grid_j = 0;
    char* line;
    int nparams_per_line = 5;

    PotentialFileReader c6ab_reader(lmp, path_c6ab, "d3");

    while ((line = c6ab_reader.next_line(nparams_per_line))) {
        try {
            ValueTokenizer c6ab_values(line);
            const double ref_c6 = c6ab_values.next_double();
            int atom_number_1 = static_cast<int>(c6ab_values.next_double());
            int atom_number_2 = static_cast<int>(c6ab_values.next_double());
            get_limit_in_pars_array(atom_number_1, atom_number_2, grid_i, grid_j);
            const int idx_atom_1 = is_int_in_array(atomic_numbers, ntypes, atom_number_1);
            if ( idx_atom_1 < 0 ) { continue; }
            const int idx_atom_2 = is_int_in_array(atomic_numbers, ntypes, atom_number_2);
            if ( idx_atom_2 < 0 ) { continue; }
            const double ref_cn1 = c6ab_values.next_double();
            const double ref_cn2 = c6ab_values.next_double();

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
   Get derivative of C6 w.r.t. CN 
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Calculate CN
------------------------------------------------------------------------- */

void PairD3::gather_cn() {
    int nthreads = omp_get_max_threads();
    int n = atom->natoms;
    double **x = atom->x;
    int *type = atom->type;
    int *ilist = cn_full->ilist;
    int *numneigh = cn_full->numneigh;      // j loop cond
    int **firstneigh = cn_full->firstneigh; // j list
    for (int ii = 0; ii < n; ii++) { cn[ii] = 0; }
    #pragma omp parallel for schedule(auto)
    for (int ii = 0; ii < n; ii++) {
        const int i = ilist[ii];
        const double rcovi = rcov[type[i]];
        const int jnum = numneigh[i];
        const int *jlist = firstneigh[i];
        for (int jj = 0; jj < jnum; jj++) {
            int j = jlist[jj]; // atom over pbc is different atom
            const double rcovj = rcov[type[j]];
            const double delij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1], x[j][2] - x[i][2]};
            const double r2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
            if (r2 > sq_cn_ang) { continue; }
            cn[ii] += 1.0 / (1.0 + exp(-k1 * (((rcovi + rcovj) / (std::sqrt(r2) * ang_to_au)) - 1.0)));
        }
    }
}

void PairD3::neigh_style_force_compute() {
    int n = atom->natoms;
    int *type = atom->type;
    int *ilist = vdw_half->ilist;
    tagint *tag = atom->tag;  // tag - 1 gives 'ii' or 'iat' index (non-spatial decomposition)

    struct CoordNInfo {
        double c6;
        double dc6iji;
        double dc6ijj;
        CoordNInfo() {}
    };

    std::vector<CoordNInfo> pair_coord_info(n * n);

    #pragma omp parallel for schedule(auto)
    for (int ii = 0; ii < n; ii++) {
        const int i = ilist[ii];
        const int itag = tag[i] - 1;  // same as ii
        const int itype = type[i];
        const double cni = cn[itag];
        const int max_ci = mxc[itype];
        for (int jj = ii; jj < n; jj++) {
            // later, we should use r2 to check cutoff w.r.t vdw
            // This is the case when we use smaller cell (cell param < vdw cutoff * 2)
            const int j = ilist[jj];
            const int jtype = type[j];
            const int jtag = tag[j] - 1;
            const double cnj = cn[jtag];
            const int max_cj = mxc[jtype];

            double denominator = 0.0;
            double numerator = 0.0;
            double d_numerator_i = 0.0;
            double d_denominator_i = 0.0;
            double d_numerator_j = 0.0;
            double d_denominator_j = 0.0;
            
            for (int a = 0; a < max_ci; a++) {
                for (int b = 0; b < max_cj; b++) {
                    const double* c6ref_arr = c6ab[itype][jtype][a][b];
                    const double c6ref = c6ref_arr[0];
                    if (c6ref <= 0.0) { continue; }  // 0.0 in no ref case isnt it?

                    const double cn_refi = c6ref_arr[1];
                    const double cn_refj = c6ref_arr[2];

                    const double r =  // this is r of cn (do not confuse)
                        (cn_refi - cni) * (cn_refi - cni) + (cn_refj - cnj) * (cn_refj - cnj);
                    double expterm = std::exp(k3 * r);
                    numerator += c6ref * expterm;
                    denominator += expterm;
                    expterm *= 2.0 * k3;
                    double term = expterm * (cni - cn_refi);
                    d_numerator_i += c6ref * term;
                    d_denominator_i += term;
                    term = expterm * (cnj - cn_refj);
                    d_numerator_j += c6ref * term;
                    d_denominator_j += term;
                }
            }
            const double c6 = numerator / denominator;
            const double sq_dnomi = 1 / (denominator * denominator);
            const double dc6iji = 
                ((d_numerator_i * denominator) - (d_denominator_i * numerator)) * sq_dnomi;
            const double dc6ijj = 
                ((d_numerator_j * denominator) - (d_denominator_j * numerator)) * sq_dnomi;
            CoordNInfo& cinfoij = pair_coord_info[ii * n + jj];
            CoordNInfo& cinfoji = pair_coord_info[jj * n + ii];
            cinfoij.c6 = c6;
            cinfoij.dc6iji = dc6iji;
            cinfoij.dc6ijj = dc6ijj;
            cinfoji.c6 = c6;
            cinfoji.dc6iji = dc6ijj;
            cinfoji.dc6ijj = dc6iji;
        }
    }

    double **x = atom->x;
    int *numneigh = vdw_half->numneigh;      // j loop cond
    int **firstneigh = vdw_half->firstneigh; // j list
    double** f_lammps = atom->f;       // give it to lammps!

    const double s8 = s18;

    double disp_sum = 0.0;
    for (int ii = 0; ii < n; ii++) { dc6i[ii] = 0.0; }

    #pragma omp parallel for schedule(auto) reduction(+:disp_sum, dc6i[:n])
    for (int ii = 0; ii < n; ii++) {
        const int i = ilist[ii];
        const int itag = tag[i] - 1;  // same as ii
        const int itype = type[i];
        const int jnum = numneigh[i];
        const int *jlist = firstneigh[i];
        const double cni = cn[itag];
        const int max_ci = mxc[itype];
        for (int jj = 0; jj < jnum; jj++) {
            int j = jlist[jj]; // atom over pbc is different atom
            const double delij[3] = {(x[j][0] - x[i][0]) / au_to_ang, 
                                     (x[j][1] - x[i][1]) / au_to_ang, 
                                     (x[j][2] - x[i][2]) / au_to_ang};
            const double r2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
            if (r2 > rthr) { continue; }
            const int jtype = type[j];
            const int jtag = tag[j] - 1;
            // force coeff //
            const double r = std::sqrt(r2);
            const double r0 = r0ab[itype][jtype];

            const double base6 = (rs6 * r0) / r;
            double t6 = base6;
            t6 *= t6; t6 *= base6; t6 *= t6; t6 *= base6; t6 *= t6;
            const double damp6 = 1.0 / (1.0 + 6.0 * t6);
            t6 *= damp6;

            double t8 = (rs8 * r0) / r;
            t8 *= t8; t8 *= t8; t8 *= t8; t8 *= t8;
            const double damp8 = 1.0 / (1.0 + 6.0 * t8);
            t8 *= damp8;

            const double r6 = r2 * r2 * r2;
            const double r42 = r2r4[itype] * r2r4[jtype];
            CoordNInfo& cinfoij = pair_coord_info[itag * n + jtag];
            const double x1 = 6.0 * cinfoij.c6 / (r6 * r) * (s6 * damp6 * (14.0 * t6 - 1.0) + s8 * r42 / r2 * damp8 * (48.0 * t8 - 4.0)) / r * au_to_ev / au_to_ang;
            const double vec[3] = { x1 * delij[0], x1 * delij[1], x1 * delij[2] };

            #pragma omp atomic
            f_lammps[i][0] -= vec[0];
            #pragma omp atomic
            f_lammps[i][1] -= vec[1];
            #pragma omp atomic
            f_lammps[i][2] -= vec[2];
            #pragma omp atomic
            f_lammps[j][0] += vec[0];
            #pragma omp atomic
            f_lammps[j][1] += vec[1];
            #pragma omp atomic
            f_lammps[j][2] += vec[2];

            const double dc6_rest = (s6 * damp6 + 3.0 * s8 * r42 * damp8 / r2) / r6;
            disp_sum -= dc6_rest * cinfoij.c6;  // energy
            dc6i[itag] += dc6_rest * cinfoij.dc6iji;
            dc6i[jtag] += dc6_rest * cinfoij.dc6ijj;
        }
    }
    eng_vdwl += disp_sum * au_to_ev;
}


void PairD3::neigh_style_get_force() {
    int n = atom->natoms;
    double **x = atom->x;
    int *type = atom->type;
    int *ilist = cn_half->ilist;
    int *numneigh = cn_half->numneigh;      // j loop cond
    int **firstneigh = cn_half->firstneigh; // j list
    tagint *tag = atom->tag;
    double** f_lammps = atom->f;       // Local force of atoms

    #pragma omp parallel for schedule(auto)
    for (int ii = 0; ii < n; ii++) {
        const int i = ilist[ii];
        const int itag = tag[i] - 1;
        const int itype = type[i];
        const int jnum = numneigh[i];
        const int *jlist = firstneigh[i];
        const double cni = cn[itag];
        for (int jj = 0; jj < jnum; jj++) {
            int j = jlist[jj]; // atom over pbc is different atom
            const double delij[3] = {(x[j][0] - x[i][0]) / au_to_ang, 
                                     (x[j][1] - x[i][1]) / au_to_ang, 
                                     (x[j][2] - x[i][2]) / au_to_ang};
            const double r2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
            if (r2 > cn_thr) { continue; }

            const int jtype = type[j];
            const int jtag = tag[j] - 1;
            const double r = std::sqrt(r2);
            const double rcovij = rcov[itype] + rcov[jtype];
            const double expterm = std::exp(-k1 * (rcovij / r - 1.0));
            const double dcnn = -k1 * rcovij * expterm / (r2 * (expterm + 1.0) * (expterm + 1.0));
            const double x1 = dcnn * (dc6i[itag] + dc6i[jtag]) / r * au_to_ev / au_to_ang;
            const double vec[3] = { x1 * delij[0], x1 * delij[1], x1 * delij[2] };
            #pragma omp atomic
            f_lammps[i][0] -= vec[0];
            #pragma omp atomic
            f_lammps[i][1] -= vec[1];
            #pragma omp atomic
            f_lammps[i][2] -= vec[2];
            #pragma omp atomic
            f_lammps[j][0] += vec[0];
            #pragma omp atomic
            f_lammps[j][1] += vec[1];
            #pragma omp atomic
            f_lammps[j][2] += vec[2];
        }
    }
}

/* ----------------------------------------------------------------------
   Compute : energy, force, and stress (Required)
------------------------------------------------------------------------- */

void PairD3::compute(int eflag, int vflag) {
    if (eflag || vflag) ev_setup(eflag, vflag);
    else evflag = vflag_fdotr = 0;

    int n = atom->natoms;       // Global number of atoms in the cell
    if (n >= n_save) {
        reallocate_arrays();
    }

    gather_cn();
    neigh_style_force_compute();
    neigh_style_get_force();

    if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairD3::init_one(int i, int j) {
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
    // No need to count local neighbor in D3
    /* return std::sqrt(rthr * std::pow(au_to_ang, 2)); */

    // give max cutoff, which is rthr not cn_thr
    return std::sqrt(sq_vdw_ang);
    //return 0.0;
}

/* ----------------------------------------------------------------------
   init specific to this pair style (Optional)
------------------------------------------------------------------------- */

void PairD3::init_style() {
    NeighRequest* cn_full_req = neighbor->add_request(this, NeighConst::REQ_FULL);
    cn_full_req->set_cutoff(std::sqrt(sq_cn_ang));
    cn_full_req->set_id(NEIGH_CN_FULL_ID);

    NeighRequest* cn_half_req = neighbor->add_request(this, NeighConst::REQ_DEFAULT);
    cn_half_req->set_cutoff(std::sqrt(sq_cn_ang));
    cn_half_req->set_id(NEIGH_CN_HALF_ID);

    NeighRequest* vdw_half_req = neighbor->add_request(this, NeighConst::REQ_DEFAULT);
    vdw_half_req->set_cutoff(std::sqrt(sq_vdw_ang));
    vdw_half_req->set_id(NEIGH_VDW_HALF_ID);
}

void PairD3::init_list(int which, NeighList* ptr) {
    switch (which) {
        case NEIGH_CN_FULL_ID:
            cn_full = ptr;
            break;
        case NEIGH_CN_HALF_ID:
            cn_half = ptr;
            break;
        case NEIGH_VDW_HALF_ID:
            vdw_half = ptr;
            break;
        default:
            error->all(FLERR, "Invalid neighbor list id");
    }
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
