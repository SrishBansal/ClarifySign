"""Dependency-free local wiring demo; it is not a production UI."""

from app.backend.factory import build_local_service
from clarifysign.data.schemas import SignToTextRequest


def main() -> None:
    result = build_local_service().sign_to_text(SignToTextRequest(((0.0, 1.0),)), "en")
    print(result)


if __name__ == "__main__":
    main()
