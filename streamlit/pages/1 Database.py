import warnings
import streamlit as st

warnings.simplefilter('ignore')

st.title("Garanteo | Database")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":world_map: Mapping",
    ":clock2: sessions.csv",
    ":three_button_mouse: events.csv",
    ":money_with_wings: campaigns.csv",
    ":bust_in_silhouette: prospects.csv"                                  
])

with tab1:
    st.subheader("Mapping")
    st.image("streamlit/images/database.png")

with tab2:
    st.subheader("Table *sessions*")
    st.markdown("##### Information relating to users' connections on Garanteo's website")
    col1, col2 = st.columns(2)
    col1.metric("Rows", 22973)
    col2.metric("Columns", 9)
    col1.metric("Primary key", "session_id")
    col2.metric("Related tables", "events, prospects")
    sessions_dict={
        "Fields":[
            "session_id",
            "session_started_at",
            "session_ended_at",
            "user_id",
            "device_type",
            "device_browser",
            "device_operating_system",
            "device_language",
            "country"
        ],
        "Key":[
            "Primary",
            "/",
            "/",
            "Secondary",
            "/",
            "/",
            "/",
            "/",
            "/",
        ],
        "Type":[
            "Integer",
            "Datetime",
            "Datetime",
            "Integer",
            "Varchar",
            "Varchar",
            "Varchar",
            "Varchar",
            "Varchar",
        ],
        "Not NaN":[
            20514,
            22973,
            22973,
            22973,
            21862,
            18367,
            14271,
            13898,
            20685
        ],
        "Description":[
            "Unique ID for each session on the website",
            "Date & time of session start",
            "Date & time of session end",
            "Unique ID for each user",
            "Type of the device which connected to the website",
            "Browser from which user conneccted to the website",
            "OS of the device which connected to the website",
            "Language of the device which connected to the website",
            "Country in which the device was located"
        ]
    }

    st.dataframe(sessions_dict)

with tab3:
    st.subheader("Table *events*")
    st.markdown("##### Information relating to users' actions on Garanteo's website")
    col1, col2 = st.columns(2)
    col1.metric("Rows", 88986)
    col2.metric("Columns", 9)
    col1.metric("Primary key", "event_id")
    col2.metric("Related tables", "All")

    events_dict={
        "Fields":[
            "event_id",
            "session_id",
            "event_timestamp",
            "event_type",
            "url",
            "referrer",
            "medium",
            "campaign_id",
            "user_id",
        
        ],
        "Key":[
            "Primary",
            "Foreign",
            "/",
            "/",
            "/",
            "/",
            "/",
            "Foreign",
            "Secondary",
        ],
        "Type":[
            "Varchar",
            "Integer",
            "Datetime",
            "Varchar",
            "Varchar",
            "Varchar",
            "Varchar",
            "Varchar",
            "Integer",
        ],
        "Not NaN":[
            84018,
            88986,
            88986,
            88611,
            88986,
            87137,
            88986,
            19625,
            88986
        ],
        "Description":[
            "Unique ID for each event on the website",
            "Session in which the action occurred",
            "Date & time of the action",
            "Click or CTC (Click-To-Call, i.e. callback request)",
            "Page of the website on which the event occurred",
            "Url of the page on which the user was just before",
            "Commercial origin of the page visit",
            "ID of the marketing campaign which originated the visit",
            "Unique ID for each user"
        ]
    }

    st.dataframe(events_dict)

with tab4:
    st.subheader("Table *campaigns*")
    st.markdown("##### Information relating to marketing campaigns")
    col1, col2 = st.columns(2)
    col1.metric("Rows", 11)
    col2.metric("Columns", 3)
    col1.metric("Primary key", "campaign_id")
    col2.metric("Related tables", "events")

    campaigns_dict={
        "Fields":[
            "campaign_id",
            "campaign_type",
            "total_cost"
        ],
        "Key":[
            "Primary",
            "/",
            "/"
        ],
        "Type":[
            "Varchar",
            "Varchar",
            "Integer"
        ],
        "Not NaN":[
            11,
            11,
            11
        ],
        "Description":[
            "Unique ID for each marketing campaign",
            "Google ads or Affiliation",
            "Cost of the campaign in 2023"
        ]
    }

    st.dataframe(campaigns_dict)


with tab5:
    st.subheader("Table *prospects*")
    st.markdown("##### Information relating to users who made a callback request (Click-To-Call, CTC)")
    col1, col2 = st.columns(2)
    col1.metric("Rows", 2060)
    col2.metric("Columns", 7)
    col1.metric("Primary key", "prospect_id")
    col2.metric("Related tables", "events, sessions")

    prospects_dict={
        "Fields":[
            "prospect_id",
            "user_id",
            "prospect_creation_date",
            "is_presented_prospect",
            "is_client",
            "gender",
            "age"
        ],
        "Key":[
            "Primary",
            "Secondary",
            "/",
            "/",
            "/",
            "/",
            "/"
        ],
        "Type":[
            "Varchar",
            "Integer",
            "Datetime",
            "Binary",
            "Binary",
            "Varchar",
            "Integer"
        ],
        "Not NaN":[
            2060,
            2060,
            2060,
            2060,
            2060,
            2007,
            2060
        ],
        "Description":[
            "Unique ID for each prospect",
            "User ID of the prospect",
            "Date of the first callback request (CTC), trigerring prospect onboarding",
            "Indicates whether the prospect was called back by Garanteo's sales",
            "Indicated whether the prospect converted into a client",
            "Prospect gender if indicated",
            "Prospect age"
        ]
    }

    st.dataframe(prospects_dict)