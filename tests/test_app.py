"""App flows and actual saved-model inference; no training or downloads."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import app


class ApplicationTests(unittest.TestCase):
    def test_empty_and_whitespace(self):
        for first, second in (("", "hello"), ("hello", ""), (" \n\t", "hi"), ("hi", "  ")):
            with self.subTest(pair=(first, second)), patch("app.get_predictor") as loader:
                with self.assertRaises(ValueError):
                    app.analyze(first, second, "Sentence-BERT")
                loader.assert_not_called()

    def test_text_is_preserved(self):
        from threading import Lock
        first, second = "  literal null  ", "NA\n"
        with patch("app.get_predictor") as loader:
            from unittest.mock import Mock
            predict = Mock(return_value={"label": "Paraphrase"})
            loader.return_value = predict, Lock()
            app.analyze(first, second, "TF-IDF Baseline")
            predict.assert_called_once_with(first, second)

    def test_real_tfidf(self):
        from src.predict_baseline import predict_paraphrase
        pair = app.EXAMPLES["Learning Python"]
        result = app.analyze(*pair, "TF-IDF Baseline")["TF-IDF Baseline"]
        self.assertEqual(result, predict_paraphrase(*pair))
        self.assertTrue(0 <= result["paraphrase_probability"] <= 1)

    def test_real_sbert(self):
        from src.predict_sbert import predict_paraphrase
        pair = app.EXAMPLES["Car and automobile"]
        result = app.analyze(*pair, "Sentence-BERT")["Sentence-BERT"]
        self.assertEqual(result, predict_paraphrase(*pair))
        self.assertTrue(-1 <= result["cosine_similarity"] <= 1)

    def test_real_comparison(self):
        result = app.analyze(*app.EXAMPLES["Car and automobile"], "Compare Both Models")
        self.assertEqual(set(result), set(app.OPTIONS[:2]))
        self.assertNotEqual(result["TF-IDF Baseline"]["predicted_class"], result["Sentence-BERT"]["predicted_class"])

    def test_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in app.OPTIONS[:2]:
                with self.assertRaisesRegex(FileNotFoundError, "Restore"):
                    app.check_artifacts(name, Path(directory))

    def test_ui_missing_artifact_message(self):
        import src.predict_baseline as baseline
        # Simulate a failed saved-model read without moving or changing real files.
        app.get_predictor.clear()
        with patch.object(baseline, "load_artifacts", side_effect=FileNotFoundError("Missing local artifacts: baseline_model.joblib. Restore the saved project artifacts.")):
            ui = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=60).run()
            ui.selectbox(key="model").select("TF-IDF Baseline").run()
            ui.text_area(key="sentence1").set_value("Hello")
            ui.text_area(key="sentence2").set_value("Hi")
            ui.button[1].click().run()
            self.assertFalse(ui.exception)
            self.assertTrue(any("Restore" in item.value for item in ui.error))
        app.get_predictor.clear()

    def test_ui_validation_and_no_early_inference(self):
        with patch("src.predict_sbert.predict_paraphrase") as predict:
            ui = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(ui.exception)
            self.assertEqual(ui.selectbox(key="model").value, "Sentence-BERT")
            predict.assert_not_called()
            ui.button[1].click().run()
            self.assertTrue(ui.error)
            predict.assert_not_called()

    def test_ui_example_compare_and_dashboard(self):
        ui = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=60).run()
        ui.selectbox(key="example").select("Car and automobile")
        ui.button[0].click().run()
        self.assertEqual(ui.text_area(key="sentence1").value, "He bought a car.")
        self.assertFalse(ui.metric)
        ui.selectbox(key="model").select("Compare Both Models").run()
        ui.button[1].click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(len(ui.metric), 3)
        self.assertTrue(any("disagree" in item.value for item in ui.warning))
        self.assertTrue(any("81.33%" in str(item.value) for item in ui.table))


if __name__ == "__main__":
    unittest.main()
