from typing import Any, Dict

try:
    from mathrec import recognize_strokes
except Exception:
    recognize_strokes = None


def recognize_handwriting_from_data_url(
    image_data_url: str,
    strokes=None,
    context_expr: str = ""
) -> Dict[str, Any]:

    if strokes and recognize_strokes is not None:
        try:
            print("USING MATHREC STROKE RECOGNIZER")
            return recognize_strokes(strokes)
        except Exception as e:
            return {
                "ok": False,
                "error": f"Stroke recognition failed: {e}"
            }

    return {
        "ok": False,
        "error": "No stroke data received. Please draw again."
    }