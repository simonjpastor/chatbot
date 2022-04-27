#imports
from chatterbot import ChatBot
import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
#from streamlit_chat import message
import random
import spacy
from chatterbot.trainers import ListTrainer
#from typing_extensions import Literal

nlp = spacy.load("../en_core_web_sm-3.2.0")

icon = [":fr:",":kr:",":crown:","old_key",":computer:",":desktop_computer:",":robot_face:"]
st.set_page_config(
    page_title="Jackbot",
    page_icon=f"{icon[random.randrange(0,7)]}"
)

#Instantiating Bot
bot = ChatBot(
    'Norman',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.BestMatch',
       ])

#Training Bot
trainer = ListTrainer(bot)

#financial training
trainer.train([
    'How do I apply for the fellowships listed on the Jackson website?',
    'Simply check the box on the application indicating that you would like to be considered for financial assistance. You will automatically be considered for any source of funding for which you are eligible.',
    'Does Jackson offer funding?',
    'Yes. Jackson offers generous funding on a merit basis to M.P.P. students. Awards typically range from half tuition to full tuition plus a stipend. The details of funding awards are provided at the time of admission. (Read more here) M.A.S. students are not eligible for funding from Jackson.',
    "Tell me more about Funding",
    "100% of Jackson students who requested financial aid received it, with the average merit-based scholarship being of $47.8K",
])

questions = [
    'Does Jackson Admissions offer webinars?',
    'The Institute hosts online chats for prospective students during the fall admissions season and over the summer. The webinars provide an overview of the admissions process and a Q&A session with admissions staff. Visit our admissions events page for a list of upcoming sessions. You can also view past webinars on our archive page.',
    'When is the deadline for submitting applications?',
    'We are unable to accept applications received after the deadline of 11:59pm Eastern Time on 2 January. It is strongly advised to submit applications well before the deadline in order to avoid the possibility of technical issues in accessing your application. If you are experiencing such difficulties, please send an email to Graduate.Admissions@yale.edu.',
    'I am currently an undergraduate student, may I still apply?',
    'Current undergraduate students are permitted to apply to Jackson, but unless you are applying as part of a globally-focused fellowship or similar program, we strongly recommend having 1-2 years of postgraduate work experience before applying.',
    'I have been preparing to apply for the M.A. in Global Affairs at Jackson, but can’t find reference to that degree option on the Jackson website anymore. What happened to it?',
    'As Jackson prepares to launch as a professional school in Fall 2022, the Yale Corporation has approved the Master in Public Policy (M.P.P.) degree to be awarded by the Jackson School of Global Affairs starting in 2023. This means all students applying for Fall 2022 enrollment will be applying for the Master in Public Policy in Global Affairs.',
    'There is no option to apply for the M.P.P. degree in the application. What should I choose?',
    'During the 2021-2022 academic year the Jackson Institute is still part of the Graduate School of Arts and Sciences; therefore Jackson is still using the GSAS application form. Please select the M.A. in Global Affairs option on the application form. Students admitted to the M.A. program in March 2022 will be switched to the M.P.P. upon enrollment in the fall of 2022 when Jackson officially becomes a school and is able to award its own degrees.',
    'When is the deadline for submitting applications?',
    'We are unable to accept applications received after the deadline of 11:59pm Eastern Time on 2 January. It is strongly advised to submit applications well before the deadline in order to avoid the possibility of technical issues in accessing your application. If you are experiencing such difficulties, please send an email to Graduate.Admissions@yale.edu.',
    'How can I apply for an application fee waiver, if I meet the eligibility requirements?',
    'To encourage early applications, the Jackson Institute for Global Affairs will waive the application fee for all MPP and MAS program applications received by December 1. A form is not required to receive the waiver. No preference will be given to early applications. Applicants may still apply until our deadline of January 2, but the fee will not be waived automatically for applications submitted after 11:59 pm EST on December 1. After that point students can request a fee waiver directly from the Graduate School of Arts and Sciences.',
    'Where can I find further information on how to prepare and submit my application?',
    'When preparing the application, all applicants should review the Frequently Asked Questions section of the Yale Graduate School of Arts and Sciences as well as the GSAS Admissions pages. The FAQs found on the Jackson Institute website are meant to complement any existing information on those pages.',
    'Will the Jackson admissions committee take into account academic disruptions caused by the COVID-19 pandemic?',
    'Recognizing the challenges to teaching and learning during the COVID-19 pandemic, the Provost and Deans of Yale University adopted the following principle: Yale’s admissions offices for graduate and professional schools evaluate applicants holistically and will take the significant disruptions of COVID-19 into account when reviewing students’ transcripts. In particular, we will respect decisions regarding the adoption of Credit/Fail and other grading options during this unprecedented period, whether they are made by institutions or by individual students.',
    'Do you recommend submitting more than three letters of recommendation?',
    'You should submit more than three letters of recommendation only if you strongly feel that additional recommendations would add significantly to your application.',
    'How do I demonstrate that I am already proficient in a foreign language?',
    'For native English speakers, we require you to leave Yale University with the equivalent of foreign language proficiency at the L4 level - essentially the intermediate level, ready for the advanced level. To fulfill this requirement, there are three options. You would: 1) need to show a 3rd year language course on your college transcript; 2) complete a full second year language course (L4) at Yale; or 3) take a language proficiency exam at Yale and place into L5 (advanced level). Note that you can choose to earn up to four credits (or two years) of language towards your Jackson degree if you would like to continue your language study. If you are a joint-degree candidate, you can earn up to two credits toward the Jackson degree requirements.',
    'I haven’t taken any Economics classes. May I still apply?',
    'Applicants do not need to have taken Economics in order to apply and be eligible for admission. The core Economics course does require basic knowledge in economics and calculus. A diagnostic exam before the first year will determine if students need to take a prerequisite foundational economics course in the fall of the first year to prepare themselves for this course.',
    'Do I have to speak a second language to apply?',
    'Applicants do not need to be proficient in a second language in order to apply and be eligible for admission. However, we require all admitted students to reach a level of foreign language proficiency as part of the graduation requirement.',
    'How long does it take to complete a joint degree?',
    'The joint degree would mean doing one semester less for each degree program you have applied for. This effectively reduces the total program of study by a year.',
    'How do I do a joint degree?',
    'As a joint degree candidate, a student can earn two degrees in two semesters fewer than if the degrees were pursued separately. With the exception of the M.P.P./J.D. program, this is three years. Candidates must apply, and be admitted to, each school separately. Candidates can apply simultaneously at the outset or to the second program once they have matriculated in one of the programs at Yale. Candidates are expected to spend the first year in Global Affairs and one semester of the second year in the partner program. During the final year, students register with each program for one semester, although they may take courses from both programs either term.',
    'Am I limited to taking Jackson courses for the remainder of the program?',
    'No. One of the unique features of Jackson’s program is your ability to choose from courses at any school or department from across the University. The full course catalog is available here. Jackson students also get preference for courses offered at Jackson.',
    'Are core courses required as part of the M.P.P. program?',
    'Jackson’s M.P.P. requires only four core courses: GLBL 802 - Applied Methods of Analysis (Fall, first year) GLBL 805 - Comparative Politics for Global Affairs (Fall of first or second year) GLBL 803 - History and Global Affairs (Spring, first year) GLBL 804 - Economics for Global Affairs (Spring, first year) Requires basic knowledge in economics and calculus. A diagnostic exam before the first year will determine if students need to take a prerequisite foundational economics course in the fall of the first year to prepare themselves for this course. Students are able to choose classes from across the university for the remaining terms. Please refer to the current course catalogue for further information.',
    'Are there opportunities to earn money during the academic year?',
    'Yes. Many Jackson students take advantage of the opportunity to become a Research Assistant or Teaching Assistant at Jackson or in other departments across the University. More details here',
    #'How do I apply for the fellowships listed on the Jackson website?',
    #'Simply check the box on the application indicating that you would like to be considered for financial assistance. You will automatically be considered for any source of funding for which you are eligible.',
    #'Does Jackson offer funding?',
    #'Yes. Jackson offers generous funding on a merit basis to M.P.P. students. Awards typically range from half tuition to full tuition plus a stipend. The details of funding awards are provided at the time of admission. (Read more here) M.A.S. students are not eligible for funding from Jackson.',
    'I’d like to connect with Admissions. What options are available?',
    'The Jackson Admissions Office conducts recruiting events in major cities and also hosts webinars and Visit Days in the fall. Please visit our Events page (https://jackson.yale.edu/event-type/graduate-admissions/) for a list of upcoming recruitment events and links to register. Students are also welcome to visit Jackson on their own, but please be advised that availability of Admissions Office staff may be limited.',
    'I have additional questions. What’s the best way to get in touch?',
    'If you have additional questions, please send all inquiries to Jackson.Admissions@yale.edu. Alternatively, you may fill out our web-based contact form. (https://apply.jackson.yale.edu/register/inquiry)',
    'How can I contact Jackson?',
    'If you have additional questions, please send all inquiries to Jackson.Admissions@yale.edu. Alternatively, you may fill out our web-based contact form.',
    'Hi',
    "Hi, how can I help you?",
    'Where is Yale University located?',
    'Yale University is located in New Haven, Connecticut.',
    'What programs are available?',
    'Jackson offers two graduate programs, the Master in Advanced Studies and the Master in Public Policy',
    'How many students are there at Jackson?',
    'There are about 40 M.P.P students and 8 M.A.S students per year at Jackson',
    'What is the average age of the students?',
    "The average age is 26, with ages ranging from 22 to 35",
    "What is the average professional experience of students ?",
    "Students have on average 3.8 years of work experience",
    "What is the proportion of international students?",
    "48% of Jackson students are international students",
    "What is the average GPA ?",
    "3.7 is the median GPA of Jackson students",
    #"Tell me more about Funding",
    #"100% of Jackson students who requested financial aid received it, with the average merit-based scholarship being of $47.8K",
]

#general training
trainer.train(questions)

st.image('jackson_logo.png')
st.header("Jackbot, The Jackson M.P.P Chatbot")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

random_number = random.randrange(0,58,2)
def get_text():
    #question = ["Where is Yale University located?",'How can I contact Jackson?','Are there opportunities to earn money during the academic year?','I am currently an undergraduate student, may I still apply?', 'When is the deadline for submitting applications?']
    suggestion = f"""Ask me anything! """
    input_text = st.text_input(suggestion, key="input")
    return input_text

user_input = get_text()

# def send_message():
#     output = bot.get_response(str(user_input))
#     st.session_state.past.append(str(user_input))
#     st.session_state.generated.append(str(output))
#     return st.session_state.past, st.session_state.generated

# def ask_random_question():
#     user_input = questions[random_number]
#     output = bot.get_response(str(user_input))
#     st.session_state.past.append(str(user_input))
#     st.session_state.generated.append(str(output))
#     return st.session_state.past, st.session_state.generated

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("")
with col2:
    if st.button('Send Message 🚀', key="input1") or user_input:
        output = bot.get_response(str(user_input))
        st.session_state.past.append(str(user_input))
        st.session_state.generated.append(str(output))
with col3:
    if st.button('Generate & Ask Random Question 🤔', key="input2"):
        random_input = questions[random_number]
        output = bot.get_response(str(random_input))
        st.session_state.past.append(str(random_input))
        st.session_state.generated.append(str(output))
with col4:
    st.write("")

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# #Streamlit
# bot_messages = []
# user_messages = []
# bot_initial_message = "Hi, I'm Jackbot a bot created by first-year M.P.P Jackson students to attempt to answer some of the questions you may have about the Yale Jackson School of Global Affairs!"
# bot_messages.append(bot_initial_message)

# if 'count' not in st.session_state:
#     st.session_state.count = 0
#     st.session_state.bot_messages = []
#     st.session_state.user_messages = []
#     st.session_state.bot_messages.append(bot_initial_message)

# #@st.cache(suppress_st_warning=True)
# def display_messages(count, bot_messages, user_messages):
#     for i in range(0,count):
#         message(st.session_state.bot_messages[i])
#         message(st.session_state.user_messages[i],is_user=True)
#         message(st.session_state.bot_messages[i+1])
#     return count

# text_input = st.text_input('User:')
# if st.button('Send Message'):
#     st.session_state.user_messages.append(text_input)
#     response = bot.get_response(str(text_input))
#     st.session_state.bot_messages.append(str(response))
#     #st.write(st.session_state.count)
#     #st.write(st.session_state.bot_messages)
#     st.session_state.count += 1
#     for i in range(0,st.session_state.count):
#         message(st.session_state.bot_messages[i])
#         message(st.session_state.user_messages[i],is_user=True)
#         message(st.session_state.bot_messages[i+1])
    #display_messages(st.session_state.count, st.session_state.bot_messages, st.session_state.user_messages)

### Adding Examples
user_messages_examples = ["Hi","I am currently an undergraduate student, may I still apply?","When is the deadline for submitting applications?","Does Jackson offer funding?","Thank you for your help!"]
bot_messages_examples = ["Hi there! How can I help you?","Current undergraduate students are permitted to apply to Jackson, but unless you are applying as part of a globally-focused fellowship or similar program, we strongly recommend having 1-2 years of postgraduate work experience before applying.","We are unable to accept applications received after the deadline of 11:59pm Eastern Time on 2 January. It is strongly advised to submit applications well before the deadline in order to avoid the possibility of technical issues in accessing your application. If you are experiencing such difficulties, please send an email to Graduate.Admissions@yale.edu.","Yes. Jackson offers generous funding on a merit basis to M.P.P. students. Awards typically range from half tuition to full tuition plus a stipend. The details of funding awards are provided at the time of admission. (Read more here) M.A.S. students are not eligible for funding from Jackson.","No worries! Please do not hesitate if you have any other questions 😊"]

if st.button('Display examples'):
    st.markdown("Here are some examples of the conversations you can have with JackBot")
    for i in range(0,len(user_messages_examples)):
        message(user_messages_examples[i],is_user=True)
        message(bot_messages_examples[i])
# new_user_input_ids = 0

# user_messages.append(display_messages(count)[1])
# if count == 0:
#     chat_history_ids = []
#     st.session_state.old_response = ''
# else:
#     count += 1
    
# bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if count > 1 else new_user_input_ids

if input == "":
    pass







