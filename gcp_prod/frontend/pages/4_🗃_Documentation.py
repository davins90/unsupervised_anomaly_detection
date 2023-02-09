import streamlit as st

from PIL import Image
from modules import utils

###

app_logo = Image.open("images/logo.png")
icon = Image.open("images/icon.ico")

st.set_page_config(page_title="DIA - Docs", layout="wide", page_icon=icon)

utils.add_menu_title()

st.image(app_logo)
st.markdown("# Documentation")
st.write("Useful documentation for the development of the project can be found on the [main page on Github](https://github.com/davins90/unsupervised_anomaly_detection). Additional explanatory files (power point presentation and others) are available in the [following folder](https://github.com/davins90/unsupervised_anomaly_detection/tree/master/src/docs).")

st.write("A special thought and thanks to my parents, Davide and Carmen, for their endless support and encouragement. A continuous example of love and commitment in which to find rest. \
        I hope to become like them 'when I grow up'.I also thank my brothers, Paul and Luke, sources of inspiration and passion in facing daily challenges, the best gift. \
        Special thanks to my wife Rebecca, for believing in me in facing this path, encouraging me, and putting up with me in the most difficult moments. A friend and lifelong companion. \
        I cannot but thank also a dear friend, Mike, with whom, driven by the same passion, we face innovative discussions and issues of the AI world, helping me to focus the work and make some order, a fundamental and valuable help. \
        I also want to thank my friends at VirtualB, Raffaele, Jacopo, Michi, and Fabio. They have instilled in me this beautiful profession, helping me, supporting me, and feeding my curiosity with always new points of view. An incredible gym for the mind.\
        Finally, I thank all the staff of Fourth Brain, the outstanding faculty with incredible experience, for their passion and willingness to share this wonderful knowledge. Thanks also to all the classmates I have met over these months: incredibly talented friends, driven by a healthy passion and expertise who have always motivated me, week after week. Undoubtedly having met these people is the most beautiful added value of this journey.")

st.write(" ")
st.write(" ")
st.write(" ")

st.write("Thank you all for this beautiful journey, God bless you. *Trust in the Lord with all your heart and lean not on your own understanding; in all your ways submit to him, and he will make your paths straight.Do not be wise in your own eyes; fear the Lord and shun evil. This will bring health to your body and nourishment to your bones. Honour the Lord with your wealth, with the firstfruits of all your crops; then your barns will be filled to overflowing, and your vats will brim over with new wine.*")

