"""
Main file of the application that gets called at the beginning of each session.
It specifies necessery configurations and components.
"""

import streamlit as st
from layout import sidebar


def define_page_config():
    """
    Set overall configuration for the app. Specify the layout, sidebar type etc
    """
    # st.set_page_config(layout="wide", initial_sidebar_state = 'collapsed', page_title='Fake Take', page_icon="✅")

    st.set_page_config(layout="wide", page_title='Fake Take', page_icon="✅")
    # Specify the main container of the website and its padding

    st.markdown(
        f"""
        <style>
            .appview-container .main .block-container{{
                padding-top: {1.2}rem;
                padding-right: {12}%;
                padding-left: {12}%;
                padding-bottom: {0}rem;
            }}
        </style>
        """,
            unsafe_allow_html=True,
        )

def local_css(file_name: str):
    """
    Define local css file
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def main():
    """
    Main method
    """
    define_page_config() # define page config
    local_css("./assets/style.css") # define local css
    sidebar() # define page navigation sidebar


if __name__ == '__main__':
    main()