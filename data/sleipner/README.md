# Sleipner 2019 Benchmark Model — not downloaded, and why

**Status 2026-08-05: blocked on the host, not on permission.**

Vaishak authorised accepting the CO2DataShare licence. The blocker is technical.

What worked: the CKAN API returns the real file list and the exact direct URLs.

| File | Size |
|---|---|
| `sleipner_reference_model_2019_grid.zip` | 34,709,338 |
| `velocities_trends_surfaces.zip` | 9,358,878 |
| `feeders.zip` | 3,351,791 |
| `well_data.zip` | 809,092 |
| `sleipner_plumes_boundaries.zip` | 115,781 |

Base URL:
`https://co2datashare.org/dataset/e6f67cbd-abf3-4d85-a118-ed386a994c2c/resource/<resource-id>/download/<file>`

What failed, in order:
1. `GET` the download URL → **400 Invalid request parameters**.
2. `POST` the licence form (`_csrf_token` + `license_accepted=on`) to the resource page → 200, but the response is the same page with no download link, and a later `GET` of the download URL still 400s.
3. `POST` directly to the download URL → **405 Method Not Allowed**.
4. Both URL shapes (dataset *name* and dataset *uuid*) behave identically.
5. Throughout, the host intermittently returns **502 / 503** — several attempts failed before reaching the app at all, which also blanked the CSRF token on one run.

So the download is gated behind something a scripted client is not reproducing — most likely a session flag set by the page's JS when the checkbox is ticked, and the site is too unstable to reverse-engineer further without guessing.

**The two-minute human fix:** open each resource page, tick *"I confirm that I have read and accepted this dataset's licence requirements"*, click **Direct download**, and drop the five zips in this folder.

- https://co2datashare.org/dataset/sleipner-2019-benchmark-model

**Licence note that outlives the download.** The SLEIPNER CO2 REFERENCE DATASET LICENSE is CC-BY-4.0 with two changes: the material **may not be sold**, and the licence covers all data in the dataset whether or not copyright applies. Fine for research and for a demo that cites its source. Not fine inside something Gurutva charges for. `examples/verdict_carbon_storage.py` therefore stays on published survey *parameters* with simulated observations even after these files land — the real data can strengthen a paper, not the product.
