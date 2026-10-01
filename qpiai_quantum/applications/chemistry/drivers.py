import numpy as np
from dataclasses import dataclass
from typing import TypedDict
from .mappers import FermionOperator


@dataclass
class MolecularProperties:
    """
    Data structure containing computed physical properties of a molecule.
    """

    n_electrons: int
    n_orbitals: int  # Number of spin-orbitals (qubits)
    nuclear_repulsion_energy: float
    hf_energy: float
    hf_state: list[int]
    one_body_integrals: (
        np.ndarray
    )  # Spin-orbital core integrals (n_orbitals x n_orbitals)
    two_body_integrals: np.ndarray  # Spin-orbital Coulomb integrals (n_orbitals x n_orbitals x n_orbitals x n_orbitals)

    def get_fermionic_hamiltonian(self) -> FermionOperator:
        """
        Construct the electronic Hamiltonian as a FermionOperator.

        H = E_nuc + sum_{pq} h_{pq} a_p^dagger a_q + 0.5 * sum_{pqrs} h_{pqrs} a_p^dagger a_q^dagger a_s a_r
        """
        h_ferm = FermionOperator({(): self.nuclear_repulsion_energy})

        # One-body terms
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                val = self.one_body_integrals[p, q]
                if abs(val) > 1e-12:
                    h_ferm += val * FermionOperator({((p, 1), (q, 0)): 1.0})

        # Two-body terms (physicist's notation: a_p^dagger a_q^dagger a_s a_r)
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                for r in range(self.n_orbitals):
                    for s in range(self.n_orbitals):
                        val = self.two_body_integrals[p, q, r, s]
                        if abs(val) > 1e-12:
                            term = ((p, 1), (q, 1), (s, 0), (r, 0))
                            h_ferm += 0.5 * val * FermionOperator({term: 1.0})

        return h_ferm


class _FallbackMolecule(TypedDict):
    """Scalar metadata stored for an offline molecular fallback."""

    n_electrons: int
    n_orbitals: int
    nuclear_repulsion_energy: float
    hf_energy: float
    hf_state: list[int]


# Pre-computed fallback database of molecular properties for Windows / Offline usage
_FALLBACK_DATABASE: dict[str, _FallbackMolecule] = {
    "h2": {
        "n_electrons": 2,
        "n_orbitals": 4,
        "nuclear_repulsion_energy": 0.71996899,
        "hf_energy": -1.11699900,
        "hf_state": [1, 1, 0, 0],
    },
    "lih": {
        "n_electrons": 2,  # Active space (frozen core)
        "n_orbitals": 4,  # Active spin-orbitals
        "nuclear_repulsion_energy": -6.80318181,  # E_nuc + E_core
        "hf_energy": -7.86199269,
        "hf_state": [1, 1, 0, 0],
    },
}

_FALLBACK_BOND_LENGTHS: dict[str, float] = {
    "h2": 0.735,
    "lih": 1.596,
}


def _parse_diatomic_geometry(
    geometry: str,
) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]] | None:
    """Parse a two-atom Cartesian geometry used by the fallback database."""
    atom_specs = [part.strip() for part in geometry.replace("\n", ";").split(";")]
    if len(atom_specs) != 2 or any(not part for part in atom_specs):
        return None

    atoms: list[tuple[str, np.ndarray]] = []
    for atom_spec in atom_specs:
        fields = atom_spec.split()
        if len(fields) != 4:
            return None
        try:
            coordinates = np.asarray(
                [float(value) for value in fields[1:]], dtype=float
            )
        except ValueError:
            return None
        atoms.append((fields[0].lower(), coordinates))

    return atoms[0], atoms[1]


def _resolve_precomputed_molecule(
    geometry: str,
    basis: str,
    charge: int,
    multiplicity: int,
) -> str | None:
    """Return an exact fallback key for a supported molecular configuration."""
    if basis.lower() != "sto-3g" or charge != 0 or multiplicity != 1:
        return None

    shorthand = geometry.strip().lower()
    if shorthand in _FALLBACK_BOND_LENGTHS:
        return shorthand

    atoms = _parse_diatomic_geometry(geometry)
    if atoms is None:
        return None

    (first_symbol, first_position), (second_symbol, second_position) = atoms
    symbols = tuple(sorted((first_symbol, second_symbol)))
    bond_length = float(np.linalg.norm(second_position - first_position))

    if symbols == ("h", "h") and np.isclose(
        bond_length, _FALLBACK_BOND_LENGTHS["h2"], rtol=0.0, atol=1e-8
    ):
        return "h2"
    if symbols == ("h", "li") and np.isclose(
        bond_length, _FALLBACK_BOND_LENGTHS["lih"], rtol=0.0, atol=1e-8
    ):
        return "lih"
    return None


def _get_precomputed_h2() -> MolecularProperties:
    """Construct MolecularProperties for H2 from hardcoded STO-3G spatial integrals."""
    db = _FALLBACK_DATABASE["h2"]
    n_spin = db["n_orbitals"]
    h1 = np.zeros((n_spin, n_spin))
    h2 = np.zeros((n_spin, n_spin, n_spin, n_spin))

    # Spatial 1-body (H2 at 0.735 Angstroms)
    h1_spatial = np.array([[-1.25633907, 0.0], [0.0, -0.47189601]])
    # Spatial 2-body chemist notation (pr|qs)
    h2_spatial = np.zeros((2, 2, 2, 2))
    h2_spatial[0, 0, 0, 0] = 0.67571015
    h2_spatial[1, 1, 1, 1] = 0.69857372
    h2_spatial[0, 0, 1, 1] = 0.66458173
    h2_spatial[1, 1, 0, 0] = 0.66458173
    h2_spatial[0, 1, 1, 0] = 0.18093120
    h2_spatial[1, 0, 0, 1] = 0.18093120
    h2_spatial[0, 1, 0, 1] = 0.18093120
    h2_spatial[1, 0, 1, 0] = 0.18093120

    # Convert to spin-orbitals
    for p in range(2):
        for q in range(2):
            h1[2 * p, 2 * q] = h1_spatial[p, q]
            h1[2 * p + 1, 2 * q + 1] = h1_spatial[p, q]

    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    val = h2_spatial[p, r, q, s]
                    h2[2 * p, 2 * q, 2 * r, 2 * s] = val
                    h2[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = val
                    h2[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = val
                    h2[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = val

    return MolecularProperties(
        n_electrons=db["n_electrons"],
        n_orbitals=n_spin,
        nuclear_repulsion_energy=db["nuclear_repulsion_energy"],
        hf_energy=db["hf_energy"],
        hf_state=db["hf_state"],
        one_body_integrals=h1,
        two_body_integrals=h2,
    )


def _get_precomputed_lih() -> MolecularProperties:
    """Construct MolecularProperties for LiH (frozen core active space)."""
    db = _FALLBACK_DATABASE["lih"]
    n_spin = db["n_orbitals"]
    h1 = np.zeros((n_spin, n_spin))
    h2 = np.zeros((n_spin, n_spin, n_spin, n_spin))

    # STO-3G active space one-body spatial integrals (including core potential contribution)
    h1_spatial = np.array([[-0.77319966, 0.04851187], [0.04851187, -0.35617274]])
    # STO-3G active space two-body spatial integrals (chemist notation)
    h2_spatial = np.zeros((2, 2, 2, 2))
    h2_spatial[0, 0, 0, 0] = 0.48758844
    h2_spatial[1, 1, 1, 1] = 0.33792450
    h2_spatial[0, 0, 1, 1] = 0.22372441
    h2_spatial[1, 1, 0, 0] = 0.22372441
    h2_spatial[0, 1, 1, 0] = 0.01302394
    h2_spatial[1, 0, 0, 1] = 0.01302394
    h2_spatial[0, 1, 0, 1] = 0.01302394
    h2_spatial[1, 0, 1, 0] = 0.01302394

    # Convert to spin-orbitals
    for p in range(2):
        for q in range(2):
            h1[2 * p, 2 * q] = h1_spatial[p, q]
            h1[2 * p + 1, 2 * q + 1] = h1_spatial[p, q]

    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    val = h2_spatial[p, r, q, s]
                    h2[2 * p, 2 * q, 2 * r, 2 * s] = val
                    h2[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = val
                    h2[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = val
                    h2[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = val

    return MolecularProperties(
        n_electrons=db["n_electrons"],
        n_orbitals=n_spin,
        nuclear_repulsion_energy=db["nuclear_repulsion_energy"],
        hf_energy=db["hf_energy"],
        hf_state=db["hf_state"],
        one_body_integrals=h1,
        two_body_integrals=h2,
    )


class MolecularDriver:
    """
    Molecular driver to perform Hartree-Fock calculations using PySCF,
    or fall back to a local pre-computed database on Windows / Offline.
    """

    def __init__(
        self,
        geometry: str = "H 0 0 0; H 0 0 0.735",
        basis: str = "sto-3g",
        charge: int = 0,
        multiplicity: int = 1,
    ):
        self.geometry = geometry
        self.basis = basis.lower()
        self.charge = charge
        self.multiplicity = multiplicity

    def run(self) -> MolecularProperties:
        """
        Run the Hartree-Fock calculation or load from pre-computed database.

        Returns:
            MolecularProperties containing molecular integrals and energy info.
        """
        # Resolve shorthand geometries
        geometry_map = {
            "h2": "H 0 0 0; H 0 0 0.735",
            "lih": "Li 0 0 0; H 0 0 1.596",
        }
        resolved_geom = geometry_map.get(self.geometry.lower().strip(), self.geometry)

        try:
            import pyscf
            from pyscf import gto, scf, ao2mo

            mol = gto.M(
                atom=resolved_geom,
                basis=self.basis,
                charge=self.charge,
                spin=self.multiplicity - 1,
                verbose=0,
            )
            mf = scf.RHF(mol)
            mf.kernel()

            # Active space (all orbitals)
            n_orbitals = mol.nao
            h1_spatial = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
            h2_spatial = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), n_orbitals)

            n_spin = 2 * n_orbitals
            h1 = np.zeros((n_spin, n_spin))
            h2 = np.zeros((n_spin, n_spin, n_spin, n_spin))

            # Convert spatial-orbitals to spin-orbitals
            for p in range(n_orbitals):
                for q in range(n_orbitals):
                    h1[2 * p, 2 * q] = h1_spatial[p, q]
                    h1[2 * p + 1, 2 * q + 1] = h1_spatial[p, q]

            for p in range(n_orbitals):
                for q in range(n_orbitals):
                    for r in range(n_orbitals):
                        for s in range(n_orbitals):
                            val = h2_spatial[p, r, q, s]
                            h2[2 * p, 2 * q, 2 * r, 2 * s] = val
                            h2[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = val
                            h2[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = val
                            h2[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = val

            hf_state = [1] * mol.nelectron + [0] * (n_spin - mol.nelectron)

            return MolecularProperties(
                n_electrons=mol.nelectron,
                n_orbitals=n_spin,
                nuclear_repulsion_energy=float(mol.energy_nuc()),
                hf_energy=float(mf.e_tot),
                hf_state=hf_state,
                one_body_integrals=h1,
                two_body_integrals=h2,
            )

        except ImportError as exc:
            # Fall back only when the complete molecular configuration matches
            # a precomputed entry. Never silently substitute a different molecule.
            fallback_key = _resolve_precomputed_molecule(
                geometry=self.geometry,
                basis=self.basis,
                charge=self.charge,
                multiplicity=self.multiplicity,
            )
            if fallback_key == "h2":
                return _get_precomputed_h2()
            if fallback_key == "lih":
                return _get_precomputed_lih()
            raise ImportError(
                "PySCF is unavailable and no exact precomputed chemistry dataset "
                "matches the requested geometry, basis, charge, and multiplicity. "
                "Supported fallbacks are neutral singlet H2 at 0.735 Angstrom and "
                "LiH at 1.596 Angstrom in the STO-3G basis."
            ) from exc
