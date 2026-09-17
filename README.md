# Semi-Classical Signal Analysis

Website for SCSA — the method that represents a positive signal as the spectral
density of a Schrödinger operator.

**Live site:** https://boost-inria.github.io/SCSA/

## Files

| file | purpose |
|---|---|
| `index.html` | method, four interactive demos, applications, software, collaborators |
| `publications.html` | complete SCSA publication record |
| `style.css` | shared stylesheet |
| `data.js` | precomputed demo data (see below) |
| `tools/precompute.py` | regenerates `data.js` using pyscsa |
| `tools/cscsa_patch.py` | corrected C-SCSA cost function |

## How the demos work

Nothing is computed in the browser. Every curve, spectrum and image on the site
was produced by the reference library [pyscsa](https://github.com/boost-inria/pyscsa)
— Fourier spectral operator on a 96-point grid, γ = 1/2 for the 1-D signals and
γ = 2 for the images — and serialised to `data.js`, which the page only displays.

To regenerate after a change to the method or the example signals, run
`python tools/precompute.py` and replace `data.js`.

The site has no build step and no external dependencies apart from Google Fonts;
it is plain HTML, CSS and JavaScript served directly by GitHub Pages.

## Licence

Code under the MIT licence. See `LICENSE.md`.
