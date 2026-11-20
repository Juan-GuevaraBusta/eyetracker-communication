"""El tablero CAA navega con el mismo criterio que el HTML, y la mirada cae en el botón."""

import unittest

import numpy as np

import caa_board


class TestCaaNavigation(unittest.TestCase):
    def test_inicio_tiene_las_cuatro_categorias(self):
        view = caa_board.describe(caa_board.Board())
        labels = [label for label, _cat in view["tiles"]]
        self.assertEqual(labels, [
            "QUIERO QUE ME AYUDEN",
            "SIENTO ALGO",
            "QUIERO COMUNICAR ALGO",
            "QUIERO HACER O CONOCER ALGO",
        ])
        self.assertEqual(view["footer"], "hint")

    def test_pagina_siguiente_y_anterior_apagada_en_la_primera(self):
        board = caa_board.Board()
        board.activate("open:0")
        view = caa_board.describe(board)
        self.assertEqual(view["title"], "Quiero que me ayuden")
        self.assertEqual(view["page"], "1 / 2")
        self.assertFalse(view["can_prev"])
        self.assertTrue(view["can_next"])
        board.activate("next")
        self.assertEqual(caa_board.describe(board)["page"], "2 / 2")
        self.assertFalse(caa_board.describe(board)["can_next"])
        board.activate("next")
        self.assertEqual(caa_board.describe(board)["page"], "2 / 2")

    def test_dolor_pasa_por_intensidad_y_el_no_regresa(self):
        board = caa_board.Board()
        board.activate("open:1")
        board.activate("open:0")
        board.activate("open:0")
        self.assertEqual(caa_board.describe(board)["kind"], "intensity")
        board.activate("open:2")
        view = caa_board.describe(board)
        self.assertEqual(view["kind"], "confirm")
        self.assertTrue(view["urgent"])
        self.assertIn("intensa", view["message"])
        board.activate("no")
        self.assertEqual(caa_board.describe(board)["kind"], "intensity")

    def test_falta_de_aire_llega_a_aviso_prioritario(self):
        board = caa_board.Board()
        board.activate("open:1")
        board.activate("open:3")
        board.activate("open:0")
        board.activate("yes")
        view = caa_board.describe(board)
        self.assertEqual(view["kind"], "sent")
        self.assertTrue(view["alert"])
        self.assertIn("aire", view["message"])

    def test_ayuda_cancelada_vuelve_a_la_misma_pantalla(self):
        board = caa_board.Board()
        board.activate("open:0")
        board.activate("help")
        self.assertEqual(caa_board.describe(board)["kind"], "help-ask")
        board.activate("helpNo")
        self.assertEqual(caa_board.describe(board)["title"], "Quiero que me ayuden")

    def test_cada_pagina_tiene_cuatro_frases(self):
        for name, node in caa_board.NODES.items():
            for page in node["pages"]:
                self.assertEqual(len(page), 4, name)


class TestCaaGazeHit(unittest.TestCase):
    def test_el_centro_de_la_primera_casilla_es_esa_casilla(self):
        board = caa_board.Board()
        spec = caa_board.layout(board, 1440, 900)
        tile = next(t for t in spec["targets"] if t["id"] == "open:0")
        x0, y0, x1, y1 = tile["rect"]
        self.assertEqual(caa_board.target_at(spec["targets"], (x0 + x1) / 2, (y0 + y1) / 2), "open:0")
        self.assertIsNone(caa_board.target_at(spec["targets"], 0, 0))

    def test_la_flecha_apagada_no_recibe_la_mirada(self):
        board = caa_board.Board()
        board.activate("open:0")
        spec = caa_board.layout(board, 1440, 900)
        prev = next(t for t in spec["targets"] if t["id"] == "prev")
        x0, y0, x1, y1 = prev["rect"]
        self.assertFalse(prev["on"])
        self.assertIsNone(caa_board.target_at(spec["targets"], (x0 + x1) / 2, (y0 + y1) / 2))

    def test_se_dibuja_sin_romper_el_lienzo(self):
        canvas = np.full((900, 1440, 3), 255, dtype=np.uint8)
        board = caa_board.Board()
        caa_board.render_board(canvas, board, "open:1")
        self.assertEqual(canvas.shape, (900, 1440, 3))
        self.assertFalse(np.array_equal(canvas[450, 1000], (255, 255, 255)))
