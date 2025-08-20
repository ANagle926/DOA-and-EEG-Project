# versions_check.py
import sys
from importlib.metadata import version, PackageNotFoundError

def v(dist_name: str):
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None

def main():
    print(f"Python       : {sys.version.split()[0]}")
    for label, dist in [
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("scikit-learn", "scikit-learn"),
        ("Matplotlib", "matplotlib"),
        ("TensorFlow", "tensorflow"),
        ("Keras", "keras"),
    ]:
        ver = v(dist)

        # Fallback: if standalone Keras isn't installed, try tf.keras
        if label == "Keras" and ver is None:
            try:
                import tensorflow as tf  # heavy import only if needed
                ver = getattr(getattr(tf, "keras", None), "__version__", tf.__version__)
                label = "Keras (tf.keras)"
            except Exception:
                pass

        print(f"{label:<12}: {ver if ver else 'not installed'}")

if __name__ == "__main__":
    main()
