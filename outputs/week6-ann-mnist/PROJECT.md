# Week 6: Lab ANN, MNIST + Deploy

## Objective and scope
Complete the lab at https://moodle.swu.ac.th/mod/assign/view.php?id=189818, including its required ANN/MNIST implementation, deployment, and deliverables. Exact requirements must be read after Moodle sign-in; the title alone is insufficient to define them.

## Shared workspace and coordination
Shared project folder: /Users/narawich/Documents/Codex/2026-09-22/https-moodle-swu-ac-th-mod/outputs/week6-ann-mnist
This document is the single source of truth. Every participating agent must read it before work and update its relevant status, files, decisions, and verification before handoff. The lead owns planning, integration, conflict resolution, and final review. Delegate only bounded tasks with minimal required context to smaller capable models after requirements are available.

## Acceptance criteria
- Retrieve and record the full assignment instructions and any starter materials.
- Implement all specified lab requirements and produce reproducible artifacts.
- Execute appropriate training, evaluation, and deployment verification; record actual results.
- Review deliverables against the assignment rubric.
- Determine required submission format from Moodle.

## Current status and completed work
- Created shared project folder and this coordination document.
- Opened the assignment URL; Moodle redirected to the login page.
- Available browser is the Codex in-app browser; no signed-in alternate browser was available.
- Implementation and delegation have not started because assignment instructions are inaccessible.

## Relevant files and decisions
- PROJECT.md: shared project record.
- Do not infer architecture, framework, deployment destination, or submission format from the assignment title.
- Do not fabricate model metrics or deployment results.

## Remaining tasks
1. User signs into SWU Moodle in the open browser.
2. Read assignment, rubric, and starter files; refine acceptance criteria.
3. Assign bounded implementation/verification tasks if useful.
4. Integrate, run, deploy as required, and review results.
5. Package required deliverables and handle submission within user authorization.

## Verification and known issues
- Verified assignment currently resolves to https://moodle.swu.ac.th/login/index.php.
- Blocker: Moodle authentication required to read the assignment.

## User-provided resources (latest steering)
- Drive materials: https://drive.google.com/drive/folders/1CUDaaiINvVLtBqnbhTmkeTn7oHqjXgXb
- GitHub account/repositories: https://github.com/ToonRz?tab=repositories
- Next: inspect these sources to identify lab requirements and the correct repository before implementation.

## Confirmed requirements and implementation plan
- Moodle access now works. Due: 28 September 2026, 23:59 Bangkok.
- Student ID: 67102010164. Moodle display name: นรวิชญ์ มากปรางค์.
- Required folder: week6-ANN-MNIST-Deploy-67102010164. Use supplied empty Drive folder as deliverable destination.
- Required files: Intro-ANN-MNIST-67102010164.ipynb, app.py, 67102010164_mnist_model.keras, requirements.txt; copied Google Sheet NN-Random-w_67102010164.
- Notebook must include student identity, Google Sheet, GitHub repo, and deployed app links.
- Exact network: Input(28,28), Flatten, Dense(128,relu), Dense(10,softmax). Adam; sparse_categorical_crossentropy; 5 epochs; batch 32; validation split 0.2.
- Rubric: normalized MNIST and 10 labeled samples; architecture summary; training logs; test evaluation and accuracy/loss curves; classification report 0–9 and numeric/heatmap confusion matrix; correctly named model; reload and predict supplied img_1.jpg; working upload/preprocess/predict Streamlit app; valid dependencies and public Streamlit Cloud deployment; complete named deliverables and links.
- Sources: work/source/Intro-ANN-MNIST-std.ipynb and work/source/RubricsLabNN.pdf (relative to task workspace).
- GitHub accessible under ToonRz; no obvious existing MNIST repository. Starter explicitly instructs creating a new repository. Planned repository: week6-ann-mnist-deploy-67102010164.
- Lead owns notebook execution, model training, worksheet copy, cloud integration and final review. Bounded app implementation can be delegated separately.
- Prior Moodle login blocker resolved. No implementation verification yet.

## Streamlit app task
- Completed: created `app.py` and `requirements.txt` for inference with `67102010164_mnist_model.keras`.
- App behavior: cached Keras model loading; JPG/JPEG/PNG upload; default `convert("L")`, resize to `(28, 28)`, float32 `/255.0`, and batch shape `(1, 28, 28)` preprocessing; optional user-controlled polarity inversion; processed image preview; predicted digit, confidence, and all-class probability chart.
- Error handling: clear messages for absent uploads, invalid/corrupt images, missing model files, model load/prediction failures, and malformed prediction outputs.
- Dependencies target Python 3.12: TensorFlow 2.16.1, Streamlit 1.33–<2.0, Pillow 10.3–<12.0, and NumPy 1.26–<2.0.
- Verification: `app.py` passed a Python AST syntax check. End-to-end inference remains for the lead after the trained model artifact is present.

## Integration and verification completed
- Executed all 53 notebook cells successfully with zero notebook errors.
- TensorFlow 2.16.1 / Keras 3.15.1; deterministic seed 164.
- Test accuracy 0.9721000195 (9,721 / 10,000); test loss 0.0953099653.
- External supplied img_1.jpg prediction: digit 2, confidence 99.9997%.
- Reloaded saved `.keras` model and verified first 32 predictions match the trained model.
- Confusion matrix shape 10x10 and sum 10,000; notebook assertion passed.
- Streamlit app AST syntax check passed; app uses cached model load, uploader, exact grayscale/resize/255 preprocessing, prediction/confidence/probability display, and error handling.
- GitHub repository created and pushed: https://github.com/ToonRz/week6-ann-mnist-deploy-67102010164 (public, main).
- Copied worksheet created and renamed `NN-Random-w_67102010164`: https://docs.google.com/spreadsheets/d/1vlqe8af47zth2FBis0wQvH4wshqjyFTkz6NHVKW3Ug4/edit
- Supplied Drive folder renamed `week6-ANN-MNIST-Deploy-67102010164` at https://drive.google.com/drive/folders/1CUDaaiINvVLtBqnbhTmkeTn7oHqjXgXb.

## Remaining / known issues
- Streamlit Community Cloud sign-in/deploy has not been completed because it requires an interactive account authorization flow. Dashboard opened at https://share.streamlit.io.
- Drive UI accepted a directory chooser attempt but reported “file not found”; explicit individual-file upload still needs completion. Local deliverables and GitHub repo are complete. The worksheet is currently in the user's My Drive and should be moved into the renamed submission folder.
- Do not claim a public app URL until deployment is visibly verified.
