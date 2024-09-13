"""
unit test cases for the encoders file shold be written like the following

Test case for the TextOneHotEncoder class.

To create a similar test case for another class with a similar structure:
1. Create a new test case class by subclassing unittest.TestCase.
2. Add setup code in the setUp method to create an instance of the class being tested.
3. Write test methods to test the functionality of the class methods.
4. Use assertions (e.g., self.assertEqual, self.assertIsInstance) to verify the behavior of the class methods.
5. update the bin/requirements.txt file with the new dependencies neede from the class, if any new is added.

Example:

class TestMyClass(unittest.TestCase):
    def setUp(self):
        # Create an instance of the class being tested
        self.my_class_instance = MyClass()

    def test_method1(self):
        # Test method 1 of the class
        result = self.my_class_instance.method1()
        self.assertEqual(result, expected_result)

    def test_method2(self):
        # Test method 2 of the class
        result = self.my_class_instance.method2()
        self.assertIsInstance(result, expected_type)
"""

import numpy as np
import numpy.testing as npt
import unittest
from src.stimulus.data.encoding.encoders import TextOneHotEncoder, IntRankEncoder, StrClassificationIntEncoder 


class TestTextOneHotEncoderDna(unittest.TestCase):

    def setUp(self):
        self.text_encoder = TextOneHotEncoder("acgt")

    def test_encode(self):
        # Test encoding a valid sequence
        encoded_data = self.text_encoder.encode("ACGT")
        self.assertIsInstance(encoded_data, np.ndarray)
        self.assertEqual(encoded_data.shape, (4, 4))  # Expected shape for one-hot encoding of "ACGT"
        correct_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        npt.assert_array_equal(encoded_data, correct_output, "The encoded matrix is not correct")  # Make sure is elements wise correct

        # Test encoding an empty sequence
        encoded_data_out_alphabet = self.text_encoder.encode("Bubba")
        self.assertIsInstance(encoded_data_out_alphabet, np.ndarray)
        self.assertEqual(encoded_data_out_alphabet.shape, (5, 4))  # Expected shape for one-hot encoding of 5 letter word
        correct_output_out_alphabet = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        npt.assert_array_equal(encoded_data_out_alphabet, correct_output_out_alphabet, "The encoded matrix is not correct") # Make sure is elements wise correct

    def test_decode(self):
        # Test decoding a one-hot encoded sequence
        encoded_data = self.text_encoder.encode("ACGT")
        decoded_sequence = self.text_encoder.decode(encoded_data)
        self.assertIsInstance(decoded_sequence, np.ndarray)
        self.assertEqual(decoded_sequence.shape, (4, 1))  # Expected shape for the decoded sequence
        self.assertEqual("".join(decoded_sequence.flatten()), "acgt")  # Expected decoded sequence

        # Test decoding an empty one-hot encoded sequence
        encoded_data_out_alphabet = self.text_encoder.encode("Bubba")
        decoded_sequence_out_alphabet = self.text_encoder.decode(encoded_data_out_alphabet)
        self.assertIsInstance(decoded_sequence_out_alphabet, np.ndarray)
        self.assertEqual(decoded_sequence_out_alphabet.size, 5)  # Expected size for the decoded sequence

    def test_encode_all_one_element(self):
        # Test encoding a single sequence
        encoded_data = self.text_encoder.encode_all(["ACGT"])
        self.assertIsInstance(encoded_data, np.ndarray)
        self.assertEqual(encoded_data.shape, (1, 4, 4))
        correct_output = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])
        npt.assert_array_equal(encoded_data, correct_output, "The encoded matrix is not correct")  # Make sure is elements wise correct

    def test_encode_all_same_length(self):
        # Test encoding a list of sequences
        sequences = ["ACGT", "ACGT", "ACGT"]
        encoded_data = self.text_encoder.encode_all(sequences)
        self.assertIsInstance(encoded_data, np.ndarray)
        self.assertEqual(len(encoded_data), 3)
        # check the shapes within the list 
        for encoded_sequence in encoded_data:
            self.assertIsInstance(encoded_sequence, np.ndarray)
            self.assertEqual(encoded_sequence.shape, (4, 4))

    def test_encode_all_variable_length(self):
        # Test encoding a list of sequences with variable length
        sequences = ["ACGT", "ACG", "AC"]
        encoded_data = self.text_encoder.encode_all(sequences)
        self.assertIsInstance(encoded_data, list)
        self.assertEqual(len(encoded_data), 3)
        # check the shapes within the list
        self.assertEqual(encoded_data[0].shape, (4, 4))
        self.assertEqual(encoded_data[1].shape, (3, 4))
        self.assertEqual(encoded_data[2].shape, (2, 4))

class TestIntRankEncoder(unittest.TestCase):

    def setUp(self):
        self.int_encoder = IntRankEncoder()

    def test_encode(self):

        # Test encoding a list of integers
        encoded_data_list = self.int_encoder.encode_all([1, 2, 3])
        self.assertIsInstance(encoded_data_list, np.ndarray)
        self.assertEqual(encoded_data_list.shape, (3,))
        npt.assert_array_equal(encoded_data_list, np.array([0, 0.5, 1]), "The encoded list is not correct")

class TestStrClassificationIntEncoder(unittest.TestCase):

    def setUp(self):
        self.str_encoder = StrClassificationIntEncoder()

    def test_encode(self):

        # Test encoding a list of strings
        encoded_data_list = self.str_encoder.encode_all(["A", "B", "C", "A"])
        self.assertIsInstance(encoded_data_list, np.ndarray)
        self.assertEqual(encoded_data_list.shape, (4,))
        npt.assert_array_equal(encoded_data_list, np.array([0, 1, 2, 0]), "The encoded list is not correct")

if __name__ == "__main__":
    unittest.main()
