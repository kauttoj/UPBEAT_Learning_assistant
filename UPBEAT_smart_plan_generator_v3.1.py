import os
import pickle
import io
import re
import pandas as pd
import base64
import markdown2
import json
from xhtml2pdf import pisa
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from openai import OpenAI
from litellm import completion

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv('.env')
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -----------------------------
# LLM Configurations
# -----------------------------
LOCAL_MODEL = 0
if LOCAL_MODEL:
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    MODEL_STR = 'deepthinkers-phi4'  # "mradermacher/DeepThinkers-Phi4-GGUF"
    llm_config_large = {
        "model": MODEL_STR,
        "max_tokens": 10000,
        "temperature": 0.1,
    }
    llm_config_small = {
        "model": MODEL_STR,
        "temperature": 0.1,
        "max_tokens": 10000,
    }
else:
    llm_config_large = {
        "model": 'claude-3-7-sonnet-latest',
        "max_tokens": 16384,
        "temperature": 0.5,
        "thinking": {"type": "enabled", "budget_tokens": 5000}
    }
    llm_config_small = {
        "model": 'gpt-4o',
        "temperature": 0.0,
        "max_tokens": 16384
    }
    #llm_config_large=llm_config_small
# -----------------------------
# Prompt Templates
# -----------------------------
ending_text = 'We are glad to have you onboard :) __If you have any questions, please contact teachers.__'

PHASE1_PROMPT_TEMPLATE_BEGINNER = '''
# Smart learning plan (onboarding)

Welcome!

You're gearing up for the __Start Smart course__, designed to help you leverage AI tools to kickstart your business. Since we'll focus primarily on AI applications, we won't delve deeply into traditional entrepreneurship and management techniques during the course. Based on your application form, here are tailored __onboarding recommendations__ to help bridge any gaps in your entrepreneurial knowledge.

{beginner_level_materials}
<br><br>
{ending_text}
'''

PHASE1_PROMPT_TEMPLATE_ADVANCED = '''
# Smart learning plan (onboarding)

Welcome!

You're gearing up for the __Start Smart course__, designed to help you leverage AI tools to kickstart your business. Since we'll focus primarily on AI applications, we won't delve deeply into traditional entrepreneurship and management techniques during the course. 

Based on your application form, you are currently at the basic or advanced level on all training topics. Great!

{ending_text}
'''

PROMPT_MODULE_OBJECTIVES = '[detailed personalized learning objective with a clear industry focus that aligns with student background]'
PROMPT_MODULE_ASSIGNMENTS = '[detailed personalized learning assignment with clear industry focus that aligns with student background]'

PROMPT_TEMPLATE_PHASE2_4 = ''' 
# ROLE # 

You are a teacher tasked with creating a personalized Smart Learning Plan (SLP) for a young adult student to support his/her learning and entrepreneurship.

# CONTEXT # 

The training session topics are divided into four (4) core modules. Each module has general (non-personalised) objectives and assignments. The current module topic is {module_topic}.
Overall, we want to develop an entrepreneurial mindset via an integrated learning approach, which includes practical elements such as learning logs, projects, case studies, brainstorming, prototyping, testing, personal reflections, self-directed assignments, and ideation exercises. 

# STUDENT # 

The student provided the following background information is him/herself: 

<student_data> 
{student_information} 
</student_data> 

# TASK # 

We want to provide the student a personalized Smart Learning Plan to support general teaching. You must create a personalized Smart Learning Plan for the student to support him/her during the training phase. Your answer must be in Markdown format with the following structure where you must write parts inside parenthesis [...]. 

--------------- 
<internal_planning> 
[your detailed internal thinking and planning how to write the Smart Learning Plan] 
</internal_planning> 

<Smart_Learning_Plan> 
# Smart learning plan (module {module_number}) 

Hi there!

These recommendations are designed to support your preparation for module {module_number}. 

## Learning objectives 

{module_objectives} 

## Assignments 

{module_assignments}

<br><br>
We hope these recommendations will help you prepare to module {module_number}. __If you have any questions, please contact teachers.__ 
</Smart_Learning_Plan> 
--------------- 

# INSTRUCTIONS # 

Analyze the student’s provided background information to understand his/her skills, industry focus, interests and goals. Consider how the training topic of module {module_number} can support the student to reach his/her short and long-term goals. 

Final Output Structure: The final output must be written entirely in MARKDOWN, contained within <Smart_Learning_Plan> section with all planning steps explained in <planning> section.  
When writing SLP, use clear structure and bullet-points. 

Important: 
-Use the provided format of the output where you ONLY complete the parts pointed by parenthesis [...]
-This plan is targeted to support topics of MODULE {module_number} described in <core_modules>
-Do NOT include detailed timetable (e.g., specific dates) for the plan. Student studies in his/her own pace. 
-DO NOT simply copy-paste of core topics or assignments, the plan must be adapted for the student 
-Think which topics are most relevant for this particular student taken into account his preferences and business aims

Now, following all above instructions and given plan structure, write the complete personalized, short to long-term Smart Learning Plan for the student.  
Remember to use Markdown format and include the plan inside <Smart_Learning_Plan> tags.
'''

PROMPT_TEMPLATE_PHASE2_4_ADVANCED = PROMPT_TEMPLATE_PHASE2_4 # FOR NOW

PROMPT_TEMPLATE_ASSISTANT = '''
# ROLE #

You are a smart "UPBEAT Learning Assistant" whose task is to help a student in learning and building his/her skill set. You mentor and support the student in his/her studies and learning in any way you can.

# CONTEXT #

The training session topics are divided into four core modules. Each module has objectives and some pre-defined assignments. This is shown below in JSON format:

{core_modules_description}

Overall, we want to develop an entrepreneurial mindset via an integrated learning approach, which includes practical elements such as learning logs, projects, case studies, brainstorming, prototyping, testing, personal reflections, self-directed assignments, and ideation exercises. 

# STUDENT #

The student has provided the following information of himself/herself via a survey that contains 18 questions [Q1-Q18]:

<student_data>
{student_information}
</student_data>

Always remember this student information and use it to personalize your responses for the student. 
In particular, take account student skill level and student responses to questions Q16, Q17, Q18 that describe industry focus, aims and motivations.
Student has a personal learning plan (called Smart Learning Plan).

# TASK #

Your task is to help the student to his/her studies and learning. You provide personalized learning assistance and mentoring. 

# INSTRUCTIONS #

-You have access to a personal learning plan (Smart Learning Plan) of the student, which you may discuss about
-Always reply so that the student gets personalized assistance based on his/her background information with potential industry focus
-Always maintain a friendly, supportive tone that encourages self-paced learning and growth of the student

**IMPORTANT:** You can only discuss about things related to learning and studying. If student asks about some completely other topics not related to studying, learning, entrepreneurship, AI or other teaching topics, simply respond "As a learning assistant, I can only discuss about learning topics." 
'''


# -----------------------------
# Utility Functions
# -----------------------------
def read_text_file(filepath):
    """Read text file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        raise

def save_text_file(filepath,content):
    """Read text file with error handling"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error writing file {filepath}: {str(e)}")
        raise

def create_modern_happy_smiley():
    """Creates a modern flat-style happy smiley image and returns it as a Base64 data URI."""
    size = (64, 64)
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    face_color = "#FFCC00"
    outline_color = "#000000"
    eye_color = "#000000"
    highlight_color = "#FFFFFF"
    draw.ellipse((4, 4, 60, 60), fill=face_color, outline=outline_color, width=3)
    draw.ellipse((18, 22, 26, 30), fill=eye_color)
    draw.ellipse((20, 24, 22, 26), fill=highlight_color)
    draw.ellipse((38, 22, 46, 30), fill=eye_color)
    draw.ellipse((40, 24, 42, 26), fill=highlight_color)
    draw.arc((16, 31, 48, 47), start=0, end=180, fill=outline_color, width=3)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def save_pdf_file(file_path, data):
    """Saves binary PDF data to a file."""
    with open(file_path, 'wb') as pdf_file:
        pdf_file.write(data)

def markdown_to_pdf(md_text):
    """Converts Markdown text to PDF binary data with blue, clickable links."""

    # Step 1: Convert Markdown to HTML
    html_text = markdown2.markdown(md_text)

    # Step 2: Auto-detect plain-text URLs and wrap them in <a> tags.
    # This regex finds any URL starting with http(s) or ftp and wraps it.
    html_text = re.sub(
        r'((?:https?|ftp):\/\/[^\s"<]+)',
        r'<a href="\1">\1</a>',
        html_text
    )

    # Step 3: Insert inline styling into every <a> tag so that links appear blue and underlined.
    html_text = re.sub(
        r'<a href="',
        r'<a style="color:blue; text-decoration:underline;" href="',
        html_text
    )

    # Step 4: Build the full HTML with additional CSS as backup.
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 1cm;
            }}
            body {{
                font-family: 'DejaVu Sans', Arial, sans-serif;
                font-size: 12pt;
                line-height: 1.6;
                color: black;
            }}
            h1 {{ color: darkblue; text-align: center; }}
            h2 {{ color: darkred; }}
            ul {{ margin-left: 20px; }}
            a, a:link, a:visited {{
                color: blue !important;
                text-decoration: underline !important;
            }}
        </style>
    </head>
    <body>
        {html_text}
    </body>
    </html>
    """

    # Step 5: Generate the PDF using xhtml2pdf (pisa)
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(full_html), dest=pdf_buffer)
    if pisa_status.err:
        raise Exception("Error in PDF generation")
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def pattern_replace(input_text, substring, target_pattern, replacement_text):
    """
    Within each occurrence of 'substring' in input_text, replace the literal target_pattern with replacement_text.
    Returns the modified text and total count of replacements.
    """
    escaped_substring = re.escape(substring)
    escaped_target = re.escape(target_pattern)
    count = 0

    def replacer(match):
        nonlocal count
        original = match.group(0)
        replaced, sub_count = re.subn(escaped_target, replacement_text, original)
        count += sub_count
        return replaced

    output_text = re.sub(escaped_substring, replacer, input_text)
    return output_text, count


def get_llm_response(prompt, llm_config):
    """Calls the LLM API using the provided prompt and configuration."""
    has_temperature = 0 if any([y in llm_config['model'] for y in ['o1', 'o3']]) else 1
    failed_count = 0
    response = None
    while failed_count < 3:
        try:
            print(f"...calling LLM {llm_config['model']}...", end='')
            if LOCAL_MODEL:
                if LOCAL_MODEL==2:
                    response = 'DUMMY LLM CALL with prompt\n\n\n' + prompt
                else:
                    response = client.chat.completions.create(
                        model=llm_config['model'],
                        messages=[{'role': "user", 'content': prompt}],
                        temperature=llm_config['temperature'],
                    )
                    response = response.choices[0].message.content
            else:
                if has_temperature:
                    response = completion(
                        model=llm_config['model'],
                        temperature=llm_config['temperature'],
                        messages=[{'role': "user", 'content': prompt}],
                        max_tokens=llm_config['max_tokens'],
                    )
                else:
                    response = completion(
                        model=llm_config['model'],
                        messages=[{'role': "user", 'content': prompt}],
                        max_tokens=llm_config['max_tokens'],
                    )
                response = response.choices[0].message.content
            print(" success")
            break
        except Exception as ex:
            print(f" FAILED: {ex}")
            failed_count += 1
    if response is None:
        raise Exception("Failed to get LLM response after several attempts.")
    return response

def extract_plan(raw_smart_plan):
    """Extracts the plan content from within <Smart_Learning_Plan> tags."""
    start = raw_smart_plan.find('<Smart_Learning_Plan>')
    if start == -1:
        raise Exception("Missing <Smart_Learning_Plan> tag")
    plan = raw_smart_plan[start + len('<Smart_Learning_Plan>'):]
    end = plan.find('</Smart_Learning_Plan>')
    if end == -1:
        raise Exception("Missing </Smart_Learning_Plan> tag")
    return plan[:end].strip()


def create_plan_prompt(incoming_survey_data, core_modules_data, study_materials, phase=1):
    """
    Constructs the student information prompt and selects the appropriate LLM template based on phase.
    For phase1, if beginner skill gaps are detected, it injects the curated materials from study_materials
    directly into the prompt.
    Returns: (prompt, student_information_prompt, recommended_materials, student_id)
    """
    valid_rows = [k for k in incoming_survey_data.index]
    student_information_prompt = ""
    for key, value in incoming_survey_data.items():
        if key in valid_rows:
            student_information_prompt += f"{key} {value}\n"

    if phase == 1:
        # Identify beginner skill gaps from Q11 questions
        count = 0
        skill_gaps = ""
        recommended_materials_keys = []
        for key, value in incoming_survey_data.items():
            if '[options 1-3]' in key and 'Beginner' in value:
                count += 1
                skill_gaps += f"{key}\n".replace(':','')
                recommended_materials_keys.append(key)
        # If there are beginner gaps, pull the exact curated materials from study_materials
        if count > 0:
            beginner_materials_list = []
            beginner_materials_str = ''
            for k, key in enumerate(recommended_materials_keys):
                material = [y for y in study_materials.index if y in key][0]
                beginner_materials_list.append(str(study_materials.loc[material, 'English']))
                beginner_materials_str += f"\n### {k + 1} {material}  \n{beginner_materials_list[-1]}\n"
            template = PHASE1_PROMPT_TEMPLATE_BEGINNER
        else:
            beginner_materials_str = ""
            template = PHASE1_PROMPT_TEMPLATE_ADVANCED

        prompt = template.replace('{beginner_level_materials}', beginner_materials_str)
        prompt = prompt.replace('{ending_text}', ending_text)

    elif 1<phase<5:
        prompt = PROMPT_TEMPLATE_PHASE2_4.replace('{student_information}', student_information_prompt)
        prompt = prompt.replace('{module_number}', str(phase))
        prompt = prompt.replace('{module_topic}',core_modules_data['dict'][phase]['title'])

        module_content = core_modules_data['dict'][phase]

        objectives = [('### __' + x + '__: ' + PROMPT_MODULE_OBJECTIVES) for x in module_content['objectives']]
        objectives_str = '\n\n'.join(objectives)

        prompt = prompt.replace('{module_objectives} ', objectives_str)

        assignments = [('### __' + x + '__: ' + PROMPT_MODULE_ASSIGNMENTS) for x in module_content['assignments']]
        assignments_str = '\n\n'.join(assignments)

        prompt = prompt.replace('{module_assignments}', assignments_str)
    elif phase==5:
        prompt = 'NO PROMPT EXISTS YET'
    else:
        raise ValueError("Invalid phase specified.")
    # Retrieve student identifier from a key field (for example, Q1. Full Name)
    return prompt, student_information_prompt

# -----------------------------
# SmartPlanGenerator Class
# -----------------------------
class SmartPlanGenerator:
    def __init__(self, core_modules_data, study_materials, additional_courses_data, llm_config_large, llm_config_small,
                 happy_img):
        self.core_modules_data = core_modules_data
        self.study_materials = study_materials
        self.additional_courses_data = additional_courses_data
        self.llm_config_large = llm_config_large
        self.llm_config_small = llm_config_small
        self.happy_img = happy_img

    def remove_border_parentheses(self, text):
        return re.sub(r'^[\[\(\{](.*?)[\]\)\}]$', r'\1', text)

    def generate_phase_plan(self, student_data, phase):
        """
        Generates the smart plan for a given phase.
        Returns a dictionary with plan_prompt, smart_plan, pdf content, student_info, recommended_materials, and student_id.
        """
        prompt, student_info = create_plan_prompt(student_data,self.core_modules_data,self.study_materials, phase=phase)

        if '# TASK #' in prompt:  # LLM is needed
            raw_plan = get_llm_response(prompt, self.llm_config_large)
            plan_content = extract_plan(raw_plan)
        else:
            raw_plan = prompt
            plan_content = prompt

        # if phase == 1:
        # plan_content = plan_content.replace(ending_text,'')

        return {
            'plan_prompt': prompt,
            'smart_plan': plan_content,
            'student_info': student_info,
        }

# -----------------------------
# Main Processing Function
# -----------------------------
STAGES_TO_GENERATE = [1]

def main():
    # Load student survey, study materials and curated materials data
    additional_courses_data = pd.read_csv(r'data/curated_additional_materials.txt', sep='|', index_col=0)
    study_materials = pd.read_excel(r'data/beginner_materials.xlsx', index_col=0)
    core_modules_data = {'text': read_text_file(r'data/description_of_training.txt')}
    with open(r'data/description_of_training.txt', 'r', encoding='utf-8') as file:
        core_modules_data['dict'] = json.load(file)
    core_modules_data['dict'] = {(i + 1): x for i, x in enumerate(core_modules_data['dict']['modules'])}
    assert all(
        [('%i' % k) in core_modules_data['dict'][k]['title'] for k in core_modules_data['dict'].keys()]), 'bad keys!'

    incoming_survey_data = pd.read_excel(r'data/upbeat_LA_data_combined_9.4.2025_with_usenames.xlsx', header=[0],skiprows=[0]).transpose()

    selected = incoming_survey_data.iloc[:, 0].apply(lambda x: x == 1)
    incoming_survey_data = incoming_survey_data.loc[selected, :]
    incoming_survey_data = incoming_survey_data.iloc[:, 1:]
    incoming_survey_data.fillna('NaN', inplace=True)
    incoming_survey_data = incoming_survey_data.map(lambda x: x.replace('\n', ' ').strip())

    skill_strings = {'Продвинутый: 3': 'Advanced [3]', 'Базовый: 2': 'Basic [2]', 'Новичок: 1': 'Beginner [1]',
                     'Advanced: 3': 'Advanced [3]', 'Basic: 2': 'Basic [2]', 'Beginner: 1': 'Beginner [1]'}

    for k, v in skill_strings.items():
        mask = incoming_survey_data.map(lambda x: k in x)
        incoming_survey_data[mask] = v

    incoming_survey_data.index = [
        next((f"{y} skill-level [options 1-3]:" for y in study_materials.index if y in x), x)
        for x in incoming_survey_data.index
    ]
    incoming_survey_data.index = [x.strip() for x in incoming_survey_data.index]

    plan_output_path = r'learning_plans_real'

    # Ensure the output directory exists
    os.makedirs(plan_output_path, exist_ok=True)

    # Create a happy smiley image for later use
    happy_img = create_modern_happy_smiley()

    # Initialize the plan generator with study_materials included
    plan_generator = SmartPlanGenerator(core_modules_data, study_materials, additional_courses_data, llm_config_large,llm_config_small, happy_img)

    # Check if we need to process any stages
    stages_to_process = []
    if STAGES_TO_GENERATE == 'all':
        stages_to_process = [1, 2, 3, 4, 5]
    elif isinstance(STAGES_TO_GENERATE, list):
        stages_to_process = [stage for stage in STAGES_TO_GENERATE if 1 <= stage <= 5]

    if not stages_to_process:
        print("No stages configured to generate. Exiting.")
        return

    # Try to load existing study plans data
    existing_study_plans = {}
    pickle_path = os.path.join(plan_output_path, 'study_plans_data.pickle')
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                existing_study_plans = pickle.load(f)
            print(f"Loaded existing study plans for {len(existing_study_plans)} students.")
        except Exception as e:
            print(f"Error loading existing study plans: {e}")
            print("Starting with fresh study plans.")

    existing_passwords = {details.get('password') for _, details in existing_study_plans.items()}
    study_plans = existing_study_plans

    # Iterate over each student (each column represents one student's survey responses)
    for plan_k, col in enumerate(incoming_survey_data.columns):
        print(f'Processing plan {plan_k + 1} of {incoming_survey_data.shape[1]}')
        student_data = incoming_survey_data[col]

        student_username = student_data['username']
        student_password = student_data['password']

        # Check if this student already exists in our data
        student_exists = student_username in study_plans

        # Generate or get password
        if not student_exists:
            # Initialize student entry if new
            study_plans[student_username] = {
                'data': student_data,
                'username': student_username,
                'password': student_password
            }

        beginner_count = sum(['Beginner' in student_data[k] for k in student_data.keys()])
        study_plans[student_username]['beginner_count'] = beginner_count

        # Generate assistant prompt if needed
        if not student_exists or 'assistant_prompt' not in study_plans[student_username]:
            # We need student_info from phase 1 for the assistant prompt
            if student_exists and 'plan_prompt_phase1' in study_plans[student_username]:
                phase1_result = {'student_info': None}  # Will be populated below
                _, phase1_result['student_info'] = create_plan_prompt(student_data, core_modules_data, study_materials,phase=1)
            else:
                # Need to generate phase 1 to get student info
                phase1_result = plan_generator.generate_phase_plan(student_data, phase=1)

            assistant_prompt = PROMPT_TEMPLATE_ASSISTANT.replace('{student_information}', phase1_result['student_info'])
            assistant_prompt = assistant_prompt.replace('{core_modules_description}', core_modules_data['text'])
            study_plans[student_username]['assistant_prompt'] = assistant_prompt

        # Regenerate plans for selected stages only
        for stage in stages_to_process:
            print(f"Generating plan for stage {stage} for student '{student_username}'...")
            stage_result = plan_generator.generate_phase_plan(student_data, phase=stage)

            # Store the generated content
            study_plans[student_username][f'plan_prompt_phase{stage}'] = stage_result['plan_prompt']
            study_plans[student_username][f'smart_plan_phase{stage}'] = stage_result['smart_plan']

            # Generate PDF and store it
            pdf_content = markdown_to_pdf(stage_result['smart_plan'])
            study_plans[student_username][f'smart_plan_pdf_phase{stage}'] = pdf_content

            # Save individual files
            save_text_file(os.path.join(plan_output_path, f'{student_username}_plan_phase{stage}.md'),
                           study_plans[student_username][f'smart_plan_phase{stage}'])

            save_pdf_file(os.path.join(plan_output_path, f'{student_username}_plan_phase{stage}.pdf'),
                          study_plans[student_username][f'smart_plan_pdf_phase{stage}'])

        # Make sure all phases exist (even if not regenerated)
        for phase in range(1, 6):
            if f'smart_plan_phase{phase}' not in study_plans[student_username]:
                study_plans[student_username][f'plan_prompt_phase{phase}'] = ''
                study_plans[student_username][f'smart_plan_phase{phase}'] = 'Not generated yet'
                study_plans[student_username][f'smart_plan_pdf_phase{phase}'] = 'Not generated yet'

        print(f"Study plans for student '{student_username}' processed.")

    # Save the study plans to a pickle file
    with open(os.path.join(plan_output_path, 'study_plans_data.pickle'), 'wb') as f:
        pickle.dump(study_plans, f)

    # Convert to CSV
    df_rows = []
    for sid, details in study_plans.items():
        df_rows.append({
            'username': sid,
            'password': details.get('password', ''),
            'plan_prompt_phase1': details.get('plan_prompt_phase1', ''),
            'plan_prompt_phase2': details.get('plan_prompt_phase2', ''),
            'plan_prompt_phase3': details.get('plan_prompt_phase3', ''),
            'plan_prompt_phase4': details.get('plan_prompt_phase4', ''),
            'plan_prompt_phase5': details.get('plan_prompt_phase5', ''),
            'smart_plan_phase1': details.get('smart_plan_phase1', ''),
            'smart_plan_phase2': details.get('smart_plan_phase2', ''),
            'smart_plan_phase3': details.get('smart_plan_phase3', ''),
            'smart_plan_phase4': details.get('smart_plan_phase4', ''),
            'smart_plan_phase5': details.get('smart_plan_phase5', ''),
            'assistant_prompt': details.get('assistant_prompt', ''),
            'beginner_count':details.get('beginner_count',''),
        })

    df_output = pd.DataFrame(df_rows)
    df_output.to_csv(os.path.join(plan_output_path, 'study_plans_data.csv'), index=False)

    print(f"All study plans processed. Stages generated: {stages_to_process}")

if __name__ == '__main__':
    main()
