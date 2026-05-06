"""
run.py
Drug Advisor Entry point 
"""

from web import create_app

app = create_app()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Drug Advisor Web Application")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)