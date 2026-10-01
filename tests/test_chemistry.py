import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pytest
from qpiai_quantum.applications.chemistry import (
    FermionOperator,
    jordan_wigner,
    bravyi_kitaev,
    MolecularDriver,
    uccsd_ansatz,
)
from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver, VQEResult


@pytest.mark.parametrize(
    "configuration",
    [
        {"geometry": "H 0 0 0; H 0 0 1.0"},
        {"geometry": "H2", "basis": "6-31g"},
        {"geometry": "H2", "charge": 1},
        {"geometry": "H2", "multiplicity": 3},
        {"geometry": "Li 0 0 0; H 0 0 2.0"},
    ],
)
def test_fallback_requires_exact_molecular_configuration(configuration):
    with patch.dict(sys.modules, {"pyscf": None}):
        with pytest.raises(ImportError, match="no exact precomputed"):
            MolecularDriver(**configuration).run()


@pytest.mark.parametrize("geometry", ["H2", "LiH"])
def test_fallback_hamiltonian_is_hermitian(geometry):
    with patch.dict(sys.modules, {"pyscf": None}):
        properties = MolecularDriver(geometry=geometry).run()
    terms = jordan_wigner(properties.get_fermionic_hamiltonian())
    matrix = _qubit_hamiltonian_matrix(terms, properties.n_orbitals)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)


def _qubit_hamiltonian_matrix(
    terms: list[tuple[list[tuple[int, str]], float]], n_qubits: int
) -> np.ndarray:
    """Build a dense matrix from little-endian SDK Pauli terms."""
    paulis = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]).astype(complex),
    }
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for operators, coefficient in terms:
        operator_by_qubit = dict(operators)
        term_matrix = np.array([[1.0]], dtype=complex)
        for qubit in reversed(range(n_qubits)):
            term_matrix = np.kron(
                term_matrix, paulis[operator_by_qubit.get(qubit, "I")]
            )
        matrix += coefficient * term_matrix
    return matrix


class TestChemistryModule(unittest.TestCase):
    def test_fermion_operator_creation(self):
        op = FermionOperator({((0, 1), (0, 0)): 1.5})
        self.assertEqual(len(op.terms), 1)
        self.assertIn(((0, 1), (0, 0)), op.terms)
        self.assertEqual(op.terms[((0, 1), (0, 0))], 1.5)

    def test_fermion_operator_arithmetic(self):
        # H = 1.5 * a0^dagger + 0.5 * a1
        op1 = FermionOperator({((0, 1),): 1.5})
        op2 = FermionOperator({((1, 0),): 0.5})

        # Addition
        op_sum = op1 + op2
        self.assertIn(((0, 1),), op_sum.terms)
        self.assertIn(((1, 0),), op_sum.terms)
        self.assertAlmostEqual(op_sum.terms[((0, 1),)], 1.5)
        self.assertAlmostEqual(op_sum.terms[((1, 0),)], 0.5)

        # Multiplication: (1.5 * a0^dagger) * (0.5 * a1) = 0.75 * a0^dagger a1
        op_mul = op1 * op2
        self.assertIn(((0, 1), (1, 0)), op_mul.terms)
        self.assertAlmostEqual(op_mul.terms[((0, 1), (1, 0))], 0.75)

    def test_jordan_wigner_mapping(self):
        # Number operator n_0 = a0^dagger a0 -> 0.5 * (I - Z0)
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        pauli_terms = jordan_wigner(op)

        # We expect: 0.5 * I - 0.5 * Z0
        terms_dict = {tuple(p): val for p, val in pauli_terms}
        self.assertIn((), terms_dict)
        self.assertAlmostEqual(terms_dict[()], 0.5)
        self.assertIn(((0, "Z"),), terms_dict)
        self.assertAlmostEqual(terms_dict[((0, "Z"),)], -0.5)

    def test_bravyi_kitaev_mapping(self):
        # Test number operator n_0 under Bravyi-Kitaev (which is also 0.5 * I - 0.5 * Z0)
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        pauli_terms = bravyi_kitaev(op, n_qubits=2)

        terms_dict = {tuple(p): val for p, val in pauli_terms}
        self.assertIn((), terms_dict)
        self.assertAlmostEqual(terms_dict[()], 0.5)
        self.assertIn(((0, "Z"),), terms_dict)
        self.assertAlmostEqual(terms_dict[((0, "Z"),)], -0.5)

    def test_molecular_driver_fallback(self):
        # Force the notebook's exact H2 configuration through the offline fallback.
        driver = MolecularDriver(
            geometry="H 0.0 0.0 0.0; H 0.0 0.0 0.735",
            basis="sto-3g",
        )
        with patch.dict(sys.modules, {"pyscf": None}):
            properties = driver.run()

        self.assertEqual(properties.n_electrons, 2)
        self.assertEqual(properties.n_orbitals, 4)
        self.assertAlmostEqual(properties.nuclear_repulsion_energy, 0.71996899)
        self.assertAlmostEqual(properties.hf_energy, -1.11699900)
        self.assertEqual(properties.hf_state, [1, 1, 0, 0])

        # Test core Hamiltonian integrals shape
        self.assertEqual(properties.one_body_integrals.shape, (4, 4))
        self.assertEqual(properties.two_body_integrals.shape, (4, 4, 4, 4))

    def test_molecular_driver_rejects_unsupported_fallback(self):
        with patch.dict(sys.modules, {"pyscf": None}):
            with self.assertRaisesRegex(
                ImportError, "no exact precomputed chemistry dataset"
            ):
                MolecularDriver(
                    geometry="He 0 0 0",
                    basis="sto-3g",
                ).run()

    def test_h2_hamiltonian_matches_same_basis_fci_reference(self):
        properties = MolecularDriver(geometry="H2", basis="sto-3g").run()
        terms = jordan_wigner(properties.get_fermionic_hamiltonian())
        matrix = _qubit_hamiltonian_matrix(terms, properties.n_orbitals)

        two_electron_basis = [
            index
            for index in range(2**properties.n_orbitals)
            if index.bit_count() == properties.n_electrons
        ]
        fci_energy = np.linalg.eigvalsh(
            matrix[np.ix_(two_electron_basis, two_electron_basis)]
        )[0]

        self.assertAlmostEqual(matrix[3, 3].real, properties.hf_energy, places=8)
        self.assertAlmostEqual(matrix[3, 12].real, 0.18093120, places=8)
        self.assertAlmostEqual(fci_energy, -1.137306039, places=8)

    def test_uccsd_ansatz_compilation(self):
        # Generate UCCSD ansatz for H2 (4 orbitals, 2 electrons)
        ansatz = uccsd_ansatz(n_qubits=4, n_electrons=2, mapping="jordan_wigner")
        self.assertEqual(ansatz.num_qubits, 4)

        # Check that it has parametric gates
        vqe = VQESolver(n_qubits=4, ansatz=lambda n: ansatz)
        n_params = vqe.num_parameters
        self.assertEqual(n_params, 3)

    @pytest.mark.skipif(
        os.environ.get("RUN_ALGO_CORRECTNESS") != "1" or not bool(os.getenv("API_KEY")),
        reason="Set RUN_ALGO_CORRECTNESS=1 and API_KEY to run authenticated local VQE integration.",
    )
    @pytest.mark.usefixtures("authenticated_sdk_user")
    def test_uccsd_statevector_optimization_respects_variational_bounds(self):
        """Verify variational bounds through the authenticated public VQE path."""
        properties = MolecularDriver(geometry="H2", basis="sto-3g").run()
        qubit_ham_terms = jordan_wigner(properties.get_fermionic_hamiltonian())

        class ChemistryHamiltonian:
            def get_hamiltonian_terms(
                self,
            ) -> list[tuple[list[tuple[int, str]], float]]:
                return qubit_ham_terms

        ansatz = uccsd_ansatz(n_qubits=4, n_electrons=2, mapping="jordan_wigner")
        vqe = VQESolver(
            n_qubits=4,
            ansatz=lambda n: ansatz,
            optimizer="cobyla",
            max_iterations=40,
            initial_point=np.zeros(3),
            verbose=False,
        )
        result = vqe.run(
            hamiltonian=ChemistryHamiltonian(),
            device_name="QpiAI-QSV-Local",
        )

        matrix = _qubit_hamiltonian_matrix(qubit_ham_terms, properties.n_orbitals)
        two_electron_basis = [
            index
            for index in range(2**properties.n_orbitals)
            if index.bit_count() == properties.n_electrons
        ]
        fci_energy = np.linalg.eigvalsh(
            matrix[np.ix_(two_electron_basis, two_electron_basis)]
        )[0]

        self.assertIsInstance(result, VQEResult)
        self.assertEqual(result.metadata["method"], "statevector")
        self.assertGreaterEqual(result.optimal_energy, fci_energy - 1e-10)
        self.assertLessEqual(result.optimal_energy, properties.hf_energy + 1e-10)


if __name__ == "__main__":
    unittest.main()
