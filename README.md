# Parse the GUI Odyssey dataset into FiftyOne Format


<img src="gui_odyssey.gif">

⬆️ Test split shown above, but this also represents the train split.

## These datasets have already been parsed into FiftyOne format and are hosted on Hugging Face


## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the train set
train_dataset = load_from_hub("Voxel51/gui-odyssey-train")

# Load the test set
test_dataset = load_from_hub("Voxel51/gui-odyssey-test")

# Launch the App
session = fo.launch_app(test_dataset)
```

# Dataset Details

## Dataset Description
- **Curated by:** OpenGVLab, Shanghai AI Laboratory, The University of Hong Kong, Nanjing University, Harbin Institute of Technology (Shenzhen), and Shanghai Jiao Tong University. Primary researchers include Quanfeng Lu, Wenqi Shao (Project Lead), Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo.
- **Funded by:** Partially supported by the National Key R & D Program of China No.2022ZD0160101 & No.2022ZD0161000.
- **Shared by:** OpenGVLab
- **Language(s) (NLP):** en
- **License:** CC BY 4.0

## Dataset Sources
- **Repository:** https://github.com/OpenGVLab/GUI-Odyssey and https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey
- **Paper:** Lu, Q., Shao, W., Liu, Z., Meng, F., Li, B., Chen, B., Huang, S., Zhang, K., Qiao, Y., & Luo, P. (2024). GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices. arXiv:2406.08451v1

# Uses

## Direct Use
- Training and evaluating cross-app GUI navigation agents for mobile devices
- Benchmarking agent performance on complex workflows requiring multiple app interactions
- Researching user experience improvements for individuals with physical disabilities
- Studying patterns in cross-app task completion

## Out-of-Scope Use
- Simulating actual financial transactions or payments
- Accessing or managing personal/private information
- Automating actions that could violate app terms of service
- Training agents for malicious GUI interactions

# Dataset Structure

The dataset contains 7,735 episodes with the following characteristics:
- **Format:** Each episode consists of a sequence of screenshots and corresponding actions
- **Average steps per episode:** 15.4 steps (significantly higher than previous datasets)
- **Apps coverage:** 201 unique applications, 1,399 app combinations
- **Device types:** 6 different mobile devices (various Pixel models including phones, tablets, foldables)
- **Action types:** 9 distinct actions (CLICK, SCROLL, LONG PRESS, TYPE, COMPLETE, IMPOSSIBLE, HOME, BACK, RECENT)
- **Metadata:** Each episode includes device information, task category, app names, and detailed action coordinates

Episodes are organized into 6 task categories with the following distribution:
- General Tool (24%)
- Information Management (18%) 
- Web Shopping (7%)
- Media Entertainment (14%)
- Social Sharing (17%)
- Multi-Apps (20%)

## FiftyOne Dataset Structure

# GUI Odyssey Train Dataset Structure

**Core Fields:**

- `episode_id`: StringField - Unique identifier for interaction sequence
- `device_name`: EmbeddedDocumentField(Classification) - Mobile device type (e.g., "Pixel Tablet")
- `step`: IntField - Sequential position within episode (zero-indexed)
- `category`: EmbeddedDocumentField(Classification) - Task category (e.g., "Social_Sharing")
- `meta_task`: EmbeddedDocumentField(Classification) - Template task pattern with placeholders
- `task`: StringField - Specific instance of meta-task with filled-in details
- `instruction`: StringField - Detailed rephrasing of task with specific applications
- `apps_used`: EmbeddedDocumentField(Classifications) - List of applications used in task
- `structured_history`: ListField(DictField) - Previous actions in structured format:
  - `step`: Step number
  - `action`: Action type (e.g., "CLICK", "SCROLL")
  - `info`: Coordinates or special values for action
- `action_points`: EmbeddedDocumentField(Keypoints) - Point-based interaction:
  - `label`: Action type (e.g., "CLICK")
  - `points`: a list of (x, y) interaction point in `[0, 1] x [0, 1]` 
- `action_type`: EmbeddedDocumentField(Classification) - General action classification
- `action_press`: EmbeddedDocumentField(Classification) - Press action details
- `action_end`: EmbeddedDocumentField(Classification) - End action details
- `action_scroll`: EmbeddedDocumentField(Polylines) - Scroll action trajectory - a list of lists of (x, y) points in `[0, 1] x [0, 1]` which are the vertices of the start, end of the scroll

# Dataset Creation

## Curation Rationale
The dataset was created to address a significant gap in existing GUI navigation datasets, which primarily focus on single-app tasks. Real-world mobile usage often requires navigating across multiple applications to complete complex tasks, such as sharing content between platforms or coordinating information between different services. GUI Odyssey specifically targets these cross-app interactions to enable more realistic and practical agent development.

## Source Data

### Data Collection and Processing
- **Collection platform:** Android Studio emulator with Android Device Bridge (ADB)
- **Process:** Human demonstrators completed tasks step-by-step following specific instructions
- **Recording:** Screenshots were saved before each action, with exact coordinates and input text recorded
- **Quality assurance:** Episodes underwent rigorous quality checks for accuracy and completeness
- **Task generation:** Instructions were created through collaboration between researchers and GPT-4 to ensure diversity

### Who are the source data producers?
The source data producers are the paper co-authors who performed the tasks on Android emulators. They received training on proper annotation procedures before data collection began.

## Annotations

### Annotation process
1. Task instructions were generated using templates with variable items and apps
2. Human annotators executed the tasks on Android emulators
3. Screenshots were automatically captured before each action
4. Action metadata was recorded, including coordinates, text input, and action type
5. Quality checks were performed to ensure consistency and accuracy

### Who are the annotators?
All co-authors of the paper participated in the annotation process after receiving training on the annotation procedure. This ensured knowledgeable annotation with consistent quality.

## Personal and Sensitive Information
The authors implemented privacy safeguards during data collection:
- Temporary accounts were used for app registrations
- No personal information was input into any applications
- The dataset does not contain any authentic personal information
- All participants provided informed consent for data inclusion

# Bias, Risks, and Limitations

- **Simulation limitations:** Certain operations like actual payments and photo-taking cannot be completed in the simulator
- **Device constraints:** The dataset only covers Google-manufactured devices due to Android Studio limitations
- **Task representation:** For complex tasks with multiple possible approaches, only one solution path is captured
- **Evaluation environment:** The dataset is currently evaluated in an offline environment, which may not fully reflect real-world performance
- **Simplified tasks:** Some real-world tasks were simplified for feasibility in data collection

# Recommendations
- Users should be aware of the platform limitations (Google devices only) when applying agents to other manufacturer devices
- Researchers should consider that captured paths represent only one of potentially many valid solutions
- When implementing agents based on this dataset, proper security measures should be implemented for sensitive operations
- For evaluation, consider both the offline metrics provided and potential online testing for comprehensive assessment

# Citation

## BibTeX:
```bibtex
@article{lu2024gui,
  title={GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices},
  author={Lu, Quanfeng and Shao, Wenqi and Liu, Zitao and Meng, Fanqing and Li, Boxuan and Chen, Botong and Huang, Siyuan and Zhang, Kaipeng and Qiao, Yu and Luo, Ping},
  journal={arXiv preprint arXiv:2406.08451},
  year={2024}
}

```

## APA:
Lu, Q., Shao, W., Liu, Z., Meng, F., Li, B., Chen, B., Huang, S., Zhang, K., Qiao, Y., & Luo, P. (2024). GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices. arXiv preprint arXiv:2406.08451.

# Dataset Card Contact
For questions about the dataset, contact the research team at OpenGVLab via the HF Dataset repository: https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey
