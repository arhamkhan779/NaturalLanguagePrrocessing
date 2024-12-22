from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    action = data.get('action')

    if action == "timings":
        response = "Our clinic is open at the following times:\nMonday to Friday: 9:00 AM to 6:00 PM\nSaturday: 10:00 AM to 4:00 PM\nSunday: Closed"
        options = [
            { "text": "Back to Main Menu", "action": "main_menu" }
        ]

    elif action == "services":
        response = "We offer the following services:\n- General Consultation\n- Pediatrics\n- Cardiology\n- Dermatology\n- Orthopedics"
        options = [
            { "text": "Back to Main Menu", "action": "main_menu" }
        ]

    elif action == "book_appointment":
        response = "To book an appointment, please call us at +1-800-555-CLINIC or visit our website for online booking."
        options = [
            { "text": "Back to Main Menu", "action": "main_menu" }
        ]

    elif action == "exit":
        response = "Thank you for using the Medical Clinic Assistant Chatbot. Have a great day!"
        options = []

    else:  # Main menu
        response = "How can I assist you today?"
        options = [
            { "text": "Clinic Timings", "action": "timings" },
            { "text": "Services Offered", "action": "services" },
            { "text": "Book an Appointment", "action": "book_appointment" },
            { "text": "Exit", "action": "exit" }
        ]

    return jsonify({ "response": response, "options": options })

if __name__ == '__main__':
    app.run(debug=True)
