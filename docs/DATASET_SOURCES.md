# Dataset sources

## INCLUDE
Official metadata/video source: AI4Bharat INCLUDE dataset. The Hugging Face dataset card documents the Zenodo record `4010759` as the video source and provides the download procedure. It also notes that INCLUDE is intended for education and isolated sign-language recognition, and that the videos were collected in Chennai.

## iSign / ISLTranslate
ISLTranslate provides approximately 31k continuous ISL-English sentence/phrase pairs. It is a research dataset for continuous translation and should be cited when discussing future continuous-sentence evaluation.

## Why the project does not bundle the videos
The complete video archives are too large to make the project practical as a normal submission ZIP. The downloader in `scripts/download_include.py` obtains them from the official source on the user's machine.
