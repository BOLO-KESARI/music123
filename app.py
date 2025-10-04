import random
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for
from markupsafe import Markup, escape
import google.generativeai as genai

app = Flask(__name__)
# Pretty-print JSON responses for easier reading when viewing raw output in browsers/tools
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Jinja filter to preserve newlines in long text responses coming from the Gemini API
def nl2br(value):
    if value is None:
        return ''
    # Escape the text first to avoid injecting HTML, then replace newlines with <br>
    return Markup('<br>').join(escape(value).split('\n'))

app.jinja_env.filters['nl2br'] = nl2br


def remove_asterisks(text):
    """Remove asterisk characters from model/API output to avoid Markdown markers in UI."""
    if text is None:
        return text
    try:
        return text.replace('*', '')
    except Exception:
        return text

# Set up Google AI API key
# Ideally, this should be an environment variable, but for this example, it's hardcoded as provided.
# Using the key from family.py and names.py for consistency.
# Prefer reading the API key from an environment variable; fall back to the existing key if not set.
AI_API_KEY = os.getenv('AI_API_KEY', "AIzaSyAtQU7XQRaC1Er4jtCI3-B-3IdbmZaXHyE")
genai.configure(api_key=AI_API_KEY)

# Create the Gemini model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

# --- Consolidated Data ---

# From family.py
RAGA_FAMILY_DATA = {
    "Bhairav Family": ["Raag Bhairav", "Raag Shree", "Raag Todi"],
    "Asavari Family": ["Raag Asavari", "Raag Marwa", "Raag Bhairavi"],
    "Bhairavi Family": ["Raag Bhairavi", "Raag Desh", "Raag Yaman Bhairavi"],
    "Dhani Family": ["Raag Dhani", "Raag Marwa", "Raag Bilawal"],
    "Todi Family": ["Raag Todi", "Raag Multani", "Raag Miyan ki Todi"],
    "Malhar Family": ["Raag Miyan ki Malhar", "Raag Malhar", "Raag Megh"],
    "Kanada Family": ["Raag Darbari Kanada", "Raag Kanada", "Raag Desh Kanada"],
    "Marwa Family": ["Raag Marwa", "Raag Purvi", "Raag Todi"],
    "Bilawal Family": ["Raag Bilawal", "Raag Yaman", "Raag Bhairav"],
    "Hamsadhwani Family": ["Raag Hamsadhwani", "Raag Shuddha Sarang"],
    "Desh Family": ["Raag Desh", "Raag Deshkar"],
    "Yaman Family": ["Raag Yaman", "Raag Yaman Kalyan"],
    "Vachaspati Family": ["Raag Vachaspati", "Raag Bahar"],
    "Lalit Family": ["Raag Lalit", "Raag Bairagi"]
}

# From general.py
general_raga_raw_data = {
    "Name": ["AhirBhairav", "Amritvarshini", "Asavari", "Bageshree", "BairagiBhairav",
             "Basant", "Bhairav", "Bhairavi", "Raga Bhatiyar Marwa That", "Bhimpalasi",
             "Bhinnashadaj", "Bihag", "Brindavani Sarang", "DarbariKanada", "Durga",
             "Kedar", "Khamaj", "Kirwani", "Lalit", "Madhuvanti", "Malkouns", "Marwa",
             "Megh", "Miamalhar", "MiyakiTodi", "Multani", "Piloo", "PooriaDhanashri",
             "Poorvi", "Shree", "SindhuBhairavi", "Sorath", "Yaman"],
    "Scale": ["S,r,G,M,P,D,n", "S,G,m,P,N", "S,R,g,M,P,d,n", "S,R,g,M,P,D,n", "S,r,M,P,n",
              "S,r,G,M,m,P,d,n", "S,r,G,M,P,d,N", "S,r,g,M,P,d,n", "S,r,G,M,m,P,D,N",
              "S,r,R,g,M,P,D,n", "S,G,M,D,N", "S,R,G,M,m,P,D,N", "S,R,M,P,n,N", "S,R,g,M,P,d,n",
              "S,R,M,P,D", "S,R,G,M,m,P,D,N", "S,R,G,M,P,D,d,n,N", "S,R,g,M,P,d,N",
              "S,r,G,M,m,d,N", "S,R,g,m,P,D,N", "S,g,M,d,n", "S,r,G,m,D,N", "S,R,M,P,n",
              "S,R,g.M,P,D,n,N", "S,r,g,m,P,d,N", "S,r,g,m,P,d,N", "S,R,g,G,M,P,d,D,n,N",
              "S,r,G,m,P,d,N", "S,R,G,M,m,P,D,N", "S,R,G,M,m,P,D,N", "S,R,M,P,D,n,N",
              "S,R,G,m,P,D,N", "S,R,G,m,P,D,N"], # Added missing scale for Yaman
    "Thaat": ["Bhairav", "Kalyan", "Asavari", "Kafi", "Bhairav", "Poorvi", "Bhairav", "Bhairavi",
              "Marwa", "Kafi", "Bilawal", "Bilawal", "Kafi", "Asavani", "Bilawal", "Kalyan",
              "Khamraj", "Mishra", "Poorvi", "Todi", "Bhairavi", "Marwa", "Kafi", "Kafi", "Todi",
              "Todi", "Kafi", "Poorvi", "Poorvi", "Poorvi", "Asavari", "Khamaj", "Kalyan"],
    "Number of Notes": ["Seven", "Five", "Seven", "Seven", "Five", "Seven", "Seven", "Seven",
                        "Seven", "Seven", "Five", "Seven", "Five", "Six", "Five", "Six", "Seven",
                        "Seven", "Five", "Seven", "Seven", "Six", "Six", "Seven", "Seven", "Seven",
                        "Seven", "Seven", "Seven", "Seven", "Seven", "Seven", "Seven"],
    "Family": ["Bhairav", "Nil", "Asavari", "Nil", "Bhairav", "Sarang", "Bhairav", "Bhairavi",
               "Nil", "Dhanashri", "Nil", "Bihag", "Sarang", "Kanada", "Nil", "Kedar", "Nil",
               "Nil", "Nil", "Nil", "Kouns", "Nil", "Malhar", "Malhar", "Todi", "Nil", "Nil",
               "Nil", "Nil", "Bhairavi", "Nil", "Nil", "Kalyan"],
    "Time": ["4a.m. - 7a.m.", "7 p.m. - 10 p.m.", "10 a.m. - 1 p.m.", "10p.m. - 1a.m.", "4a.m. - 7a.m.",
             "4a.m. - 7a.m.", "4a.m. - 7a.m.", "4a.m. - 7a.m.", "4a.m. - 7a.m.", "1a.m. - 4p.m.",
             "10p.m. - 1a.m.", "10p.m. - 1a.m.", "10a.m. - 1p.m.", "10p.m. - 1a.m.", "10p.m. - 1a.m.",
             "7p.m. - 10p.m.", "7p.m. - 10p.m.", "1a.m. -4a.m.", "4a.m. - 7a.m.", "1p.m. - 4p.m.",
             "1a.m. -4a.m.", "4p.m. - 7p.m.", "10p.m. - 1a.m.", "10p.m. - 1a.m.", "10a.m. - 1p.m.",
             "4p.m. - 7p.m.", "1p.m. - 4p.m.", "4p.m. - 7p.m.", "4p.m. - 7p.m.", "4p.m. - 7p.m.",
             "10a.m. - 1p.m.", "10p.m. - 1a.m.", "7p.m. - 10p.m."]
}
# Ensure that all columns have the same length
max_length = max(len(value) for value in general_raga_raw_data.values())
for key, value in general_raga_raw_data.items():
    if len(value) < max_length:
        value.extend([None] * (max_length - len(value)))
df_ragas = pd.DataFrame(general_raga_raw_data)


# From mood.py
raga_mood_data = {
    "Love, Attractiveness": {
        "ragas": [
            {"name": "Mohana", "description": "Sweet and romantic, with flowing melodies.", "link": "#"}, # Placeholder link
            {"name": "Kalyani", "description": "Graceful and charming, evokes beauty.", "link": "#"},
            {"name": "Kedaram", "description": "Melodic, emphasizing attraction.", "link": "#"},
            {"name": "Mand", "description": "Soothing and romantic.", "link": "#"},
            {"name": "Suruti", "description": "Delightful and engaging.", "link": "#"},
        ],
        "features": "Sweet, romantic, and graceful; these ragas emphasize melodic flow and pleasing ornamentation.",
    },
    "Laughter, Mirth, Comedy": {
        "ragas": [
            {"name": "Desh", "description": "Bright and uplifting.", "link": "#"},
            {"name": "Hamsadhwani", "description": "Light and cheerful, perfect for joyful moods.", "link": "#"},
            {"name": "Bihag", "description": "Playful and lively.", "link": "#"},
            {"name": "Hindolam", "description": "Cheerful, invoking mirth.", "link": "#"},
            {"name": "Malkauns", "description": "Majestic yet playful.", "link": "#"},
        ],
        "features": "Bright, lively, and uplifting; these ragas bring out joy and humor.",
    },
    "Fury, Anger": {
        "ragas": [
            {"name": "Marwa", "description": "Intense and dramatic.", "link": "#"},
            {"name": "Bhairavi", "description": "Expresses strong emotions with sharp transitions.", "link": "#"},
            {"name": "Darbari Kanada", "description": "Serious and intense, evoking power.", "link": "#"},
            {"name": "Hamsadhwani", "description": "Forceful and assertive.", "link": "#"},
            {"name": "Bhairav", "description": "Powerful and commanding.", "link": "#"},
        ],
        "features": "Intense, dramatic, and commanding; these ragas emphasize sharp transitions to evoke fury.",
    },
    "Compassion, Mercy": {
        "ragas": [
            {"name": "Bhairavi", "description": "Gentle and tender, full of empathy.", "link": "#"},
            {"name": "Yaman", "description": "Soothing, with a touch of sorrow.", "link": "#"},
            {"name": "Darbari Kanada", "description": "Evokes a sense of deep understanding.", "link": "#"},
            {"name": "Kafi", "description": "Soft and empathetic.", "link": "#"},
            {"name": "Kharaharapriya", "description": "Full of gentle oscillations, expressing kindness.", "link": "#"},
        ],
        "features": "Slow, tender, and empathetic; these ragas emphasize kindness and compassion.",
    },
    "Disgust, Aversion": {
        "ragas": [
            {"name": "Shivaranjani", "description": "Unsettling and tense.", "link": "#"},
            {"name": "Dhanasri", "description": "Eerie and uncomfortable.", "link": "#"},
            {"name": "Bhairavi", "description": "Tense transitions evoke aversion.", "link": "#"},
            {"name": "Darbari Kanada", "description": "Tense and dramatic.", "link": "#"},
        ],
        "features": "Unsettling and tense; these ragas evoke discomfort and aversion.",
    },
    "Horror, Terror": {
        "ragas": [
            {"name": "Shivaranjani", "description": "Dark and eerie, with an unsettling atmosphere.", "link": "#"},
            {"name": "Marwa", "description": "A sense of looming dread, intense and dramatic.", "link": "#"},
            {"name": "Darbari Kanada", "description": "Serious, ominous, and grave.", "link": "#"},
            {"name": "Bhairavi", "description": "Powerful and haunting, filled with intense emotions.", "link": "#"},
        ],
        "features": "Dark, eerie, and unsettling; these ragas evoke fear and terror.",
    },
    "Heroic Mood": {
        "ragas": [
            {"name": "Bhairav", "description": "Commanding and strong, invoking heroism.", "link": "#"},
            {"name": "Desh", "description": "Majestic and uplifting, full of grandeur.", "link": "#"},
            {"name": "Hindolam", "description": "Serene yet powerful, evoking courage.", "link": "#"},
            {"name": "Marwa", "description": "Intense and dramatic, filled with energy.", "link": "#"},
        ],
        "features": "Majestic, powerful, and uplifting; these ragas evoke heroism and courage.",
    },
    "Wonder, Amazement": {
        "ragas": [
            {"name": "Yaman", "description": "Soothing and serene, creating a sense of awe.", "link": "#"},
            {"name": "Bageshree", "description": "Mysterious and profound, evoking wonder.", "link": "#"},
            {"name": "Bihag", "description": "Intriguing and inspiring, a sense of awe.", "link": "#"},
            {"name": "Marwa", "description": "Majestic and powerful, creating wonder.", "link": "#"},
        ],
        "features": "Mysterious, awe-inspiring, and profound; these ragas evoke a sense of wonder and amazement.",
    },
    "Peace or Tranquillity": {
        "ragas": [
            {"name": "Hamsadhwani", "description": "Soft, calm, and soothing, perfect for peace.", "link": "#"},
            {"name": "Vasant", "description": "Serene and gentle, evoking a peaceful mood.", "link": "#"},
            {"name": "Shree", "description": "Calm and meditative, inducing tranquility.", "link": "#"},
            {"name": "Yaman", "description": "Gentle and serene, a perfect match for peace.", "link": "#"},
        ],
        "features": "Calm, serene, and gentle; these ragas evoke peace and tranquility.",
    },
}

# From thaat.py
thaat_data = {
    "Bhairav": ["AhirBhairav", "BairagiBhairav", "Bhairav", "Malkouns"],
    "Kalyan": ["Amritvarshini", "Kedar", "Yaman"],
    "Asavari": ["Asavari", "DarbariKanada", "SindhuBhairavi"],
    "Kafi": ["Bageshree", "Bhimpalasi", "Brindavani Sarang", "Megh", "Miamalhar", "Piloo"],
    "Poorvi": ["Basant", "Lalit", "PooriaDhanashri", "Poorvi", "Shree"],
    "Marwa": ["Raga Bhatiyar Marwa That", "Marwa"],
    "Bilawal": ["Bhinnashadaj", "Bihag", "Durga"],
    "Mishra": ["Kirwani"],
    "Todi": ["Madhuvanti", "MiyakiTodi", "Multani"],
    "Khamaj": ["Khamaj", "Sorath"],
}

# Generate a comprehensive list of raga names for the "Raga Information Finder" (replaces names.py Excel loading)
all_raga_names = set()
for ragas_list in RAGA_FAMILY_DATA.values():
    all_raga_names.update(ragas_list)
all_raga_names.update(df_ragas["Name"].dropna().tolist())
for mood_data in raga_mood_data.values():
    for raga in mood_data["ragas"]:
        all_raga_names.add(raga["name"])
for ragas_list in thaat_data.values():
    all_raga_names.update(ragas_list)
RAGA_NAMES_LIST = sorted(list(all_raga_names))


# --- Gemini API Helper Functions ---
def fetch_raga_family_details(raga_family_name):
    """Fetch additional details for a raga family using the Gemini API."""
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(f"Provide insights for the raga family {raga_family_name}.")
        # Normalize and trim whitespace for safer display
        return remove_asterisks((response.text or '').strip())
    except Exception as e:
        return f"Error fetching details for {raga_family_name}: {e}"

def fetch_raga_details_general(raga_name):
    """Fetch general insights for a raga using the Gemini API."""
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(f"Provide insights for the raga {raga_name}.")
        return remove_asterisks((response.text or '').strip())
    except Exception as e:
        return f"Error: {e}"

def fetch_raga_description_detailed(raga_name):
    """Fetch a detailed description for a raga using the Gemini API."""
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(f"Provide a detailed description of the raga {raga_name}.")
        return remove_asterisks((response.text or '').strip())
    except Exception as e:
        return f"Error fetching details for {raga_name}: {e}"

def fetch_thaat_details_for_raga(raga_name, thaat_name):
    """Fetch raga-thaat details from Gemini API."""
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(f"Provide insights for the raga '{raga_name}' in the '{thaat_name}' thaat.")
        return remove_asterisks((response.text or '').strip())
    except Exception as e:
        return f"Error fetching details: {e}"


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main Raga Prediction page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles Raga prediction from an uploaded audio file.
    This is a MOCK implementation as the actual ML model and logic are not provided.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # --- MOCK PREDICTION LOGIC ---
        # In a real application, you would:
        # 1. Save the file temporarily.
        # 2. Preprocess the audio.
        # 3. Load and run your ML model.
        # 4. Generate waveform/spectrogram images and save them to static/images.
        # 5. Return actual prediction results and image URLs.

        # For demonstration, we return mock data.
        mock_labels = ["Raag Bhairavi", "Raag Yaman", "Raag Darbari Kanada", "Raag Malkauns"]
        mock_times = ["Morning", "Evening", "Night", "Anytime"]
        mock_moods = ["Devotional", "Romantic", "Serious", "Peaceful"]

        predicted_label = random.choice(mock_labels)
        description = f"This is a mock description for {predicted_label}. It's a very expressive raga known for its deep emotional impact."
        time = random.choice(mock_times)
        mood = random.choice(mock_moods)

        waveform_url = url_for('static', filename='images/waveform_placeholder.png')
        spectrogram_url = url_for('static', filename='images/spectrogram_placeholder.png')

        return jsonify({
            'predicted_label': predicted_label,
            'description': description,
            'time': time,
            'mood': mood,
            'waveform_url': waveform_url,
            'spectrogram_url': spectrogram_url
        })
    return jsonify({'error': 'File processing failed'}), 500


@app.route('/family_insights', methods=['GET', 'POST'])
def family_insights():
    """Renders the Raga Family Insights page and handles family selection."""
    selected_family = request.form.get('selected_family') if request.method == 'POST' else None
    associated_ragas = []
    raga_family_details = None

    if selected_family and selected_family in RAGA_FAMILY_DATA:
        associated_ragas = RAGA_FAMILY_DATA[selected_family]
        raga_family_details = fetch_raga_family_details(selected_family)

    return render_template('family_insights.html',
                           raga_families=sorted(list(RAGA_FAMILY_DATA.keys())),
                           selected_family=selected_family,
                           associated_ragas=associated_ragas,
                           raga_family_details=raga_family_details)

@app.route('/raga_info', methods=['GET', 'POST'])
def raga_info():
    """Renders the General Raga Information page and handles raga selection."""
    selected_raga_name = request.form.get('selected_raga') if request.method == 'POST' else None
    selected_raga_details = None

    if selected_raga_name:
        selected_raga_row = df_ragas[df_ragas["Name"] == selected_raga_name]
        if not selected_raga_row.empty:
            selected_raga_details = selected_raga_row.iloc[0].to_dict()

    return render_template('raga_info.html',
                           raga_names=df_ragas["Name"].tolist(),
                           selected_raga_name=selected_raga_name,
                           raga_details=selected_raga_details)

@app.route('/mood_recommendation', methods=['GET', 'POST'])
def mood_recommendation():
    """Renders the Raga Recommendation by Mood page and handles mood selection."""
    selected_mood = request.form.get('selected_mood') if request.method == 'POST' else None
    ragas_for_mood = []
    mood_features = None
    raga_details_list = []

    if selected_mood and selected_mood in raga_mood_data:
        mood_data_entry = raga_mood_data[selected_mood]
        ragas_for_mood = mood_data_entry["ragas"]
        mood_features = mood_data_entry["features"]

        for raga in ragas_for_mood:
            details = fetch_raga_details_general(raga["name"])
            raga_details_list.append({"name": raga["name"], "description": raga["description"], "link": raga["link"], "details": details})

    return render_template('mood_recommendation.html',
                           moods=sorted(list(raga_mood_data.keys())),
                           selected_mood=selected_mood,
                           mood_features=mood_features,
                           raga_details_list=raga_details_list)

@app.route('/raga_finder_by_name', methods=['GET', 'POST'])
def raga_finder_by_name():
    """Renders the Raga Information Finder page and handles raga selection."""
    # The page itself simply renders the selector. Detailed descriptions are fetched
    # via the API endpoint `/api/raga_description` which returns JSON. This avoids
    # rendering raw AI text directly and prevents accidental broken output on page
    # render (the client will request JSON and render safely).
    selected_raga = request.form.get('selected_raga') if request.method == 'POST' else None
    return render_template('raga_finder_by_name.html',
                           raga_names=RAGA_NAMES_LIST,
                           selected_raga=selected_raga,
                           raga_description=None)


@app.route('/api/raga_description', methods=['POST'])
def api_raga_description():
    """Return a raga description as JSON for client-side rendering.

    Accepts application/json or form-encoded data. Returns:
      { raga: <name>, description: <string> }

    The description is the raw text returned from the Gemini helper (may contain
    newlines). The client is responsible for escaping and converting newlines to
    <br> when injecting into the DOM.
    """
    data = None
    # Try JSON first, then fallback to form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    raga = data.get('selected_raga') or data.get('raga')
    if not raga:
        return jsonify({'error': 'No raga provided'}), 400

    # Fetch the detailed description (this may call the model)
    raga_description = fetch_raga_description_detailed(raga)

    # Ensure we strip '*' characters before returning JSON to the client
    safe_description = remove_asterisks(raga_description)

    # Return as JSON so client can render safely
    return jsonify({'raga': raga, 'description': safe_description})

@app.route('/thaat_mapping', methods=['GET', 'POST'])
def thaat_mapping():
    """Renders the Raga-Thaat Mapping page and handles raga selection."""
    all_thaat_ragas = sorted(list(set(raga for ragas_list in thaat_data.values() for raga in ragas_list)))
    selected_raga = request.form.get('selected_raga') if request.method == 'POST' else None
    associated_thaat = None
    thaat_insights = None
    other_ragas_in_thaat = []

    if selected_raga:
        for thaat, ragas_list in thaat_data.items():
            if selected_raga in ragas_list:
                associated_thaat = thaat
                thaat_insights = fetch_thaat_details_for_raga(selected_raga, associated_thaat)
                other_ragas_in_thaat = [r for r in ragas_list if r != selected_raga]
                break

    return render_template('thaat_mapping.html',
                           ragas_for_select=all_thaat_ragas,
                           selected_raga=selected_raga,
                           associated_thaat=associated_thaat,
                           thaat_insights=thaat_insights,
                           other_ragas_in_thaat=other_ragas_in_thaat)

if __name__ == '__main__':
    port = random.randint(5000, 6000)
    app.run(debug=True, port=port)