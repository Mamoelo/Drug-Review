"""
Application Routes With Authentication, History, and Learning
"""

from flask import Blueprint, render_template, request, jsonify, current_app, session, redirect, url_for, flash
from web.services.predictor import PredictionService
from web.services.recommender import RecommendationService
from web.services.history_service import HistoryService
from web.services.auth_service import AuthService
from functools import wraps
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

main_bp = Blueprint('main', __name__)

SUPPORTED_CONDITIONS = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please sign in to access this page.', 'error')
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            flash('Admin access required.', 'error')
            return redirect(url_for('main.admin_login'))
        return f(*args, **kwargs)
    return decorated_function


def get_predictor():
    if not hasattr(current_app, '_predictor'):
        current_app._predictor = PredictionService(
            current_app.config['MODEL_PATH'],
            current_app.config['ENCODER_PATH']
        )
    return current_app._predictor


def get_recommender():
    if not hasattr(current_app, '_recommender'):
        current_app._recommender = RecommendationService(
            current_app.config['DATA_PATH']
        )
    return current_app._recommender


def get_history_service():
    if not hasattr(current_app, '_history'):
        current_app._history = HistoryService()
    return current_app._history


def get_auth_service():
    if not hasattr(current_app, '_auth'):
        current_app._auth = AuthService()
    return current_app._auth


@main_bp.route('/')
def index():
    recommender = get_recommender()
    conditions = recommender.get_all_conditions()
    stats = {'conditions': len(conditions)}

    user = None
    if 'user_id' in session:
        auth = get_auth_service()
        user = auth.get_user_profile(session['user_id'])

    return render_template('index.html', stats=stats, conditions=conditions, user=user)


@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        auth = get_auth_service()
        result = auth.login(email, password)

        if result['success']:
            session['user_id'] = result['user_id']
            session['user_email'] = result['email']
            session['user_name'] = result.get('full_name', email.split('@')[0])
            flash('Welcome back!', 'success')
            return redirect(url_for('main.dashboard'))

        flash(result['error'], 'error')
        return render_template('login.html')

    return render_template('login.html')


@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        age = request.form.get('age', type=int)
        gender = request.form.get('gender')

        auth = get_auth_service()
        result = auth.register(email, password, full_name, age, gender)

        if result['success']:
            flash('Account created successfully! Please sign in.', 'success')
            return redirect(url_for('main.login'))

        flash(result['error'], 'error')
        return render_template('register.html')

    return render_template('register.html')


@main_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been signed out.', 'success')
    return redirect(url_for('main.index'))


@main_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    predictor = get_predictor()
    user_id = session['user_id']
    auth = get_auth_service()
    user = auth.get_user_profile(user_id)

    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()
        # ── NEW: optional doctor's diagnosis from the form ──────────────────
        user_diagnosis = request.form.get('user_diagnosis', '').strip() or None

        if not symptoms:
            return render_template(
                'predict.html',
                error='Please describe your symptoms',
                supported_conditions=SUPPORTED_CONDITIONS,
                user=user
            )

        # Pass user_diagnosis straight into the predictor
        result = predictor.predict(symptoms, user_diagnosis=user_diagnosis)

        if result['success']:
            recommender = get_recommender()
            history_service = get_history_service()

            # ── Decide which condition to use for recommendations ───────────
            # Use the final predicted condition (already resolved by the
            # diagnosis engine inside predictor.py).
            condition = result['condition']

            # If the result is non-medical, skip recommendations
            recommendations = []
            if condition in recommender.get_all_conditions():
                base_recommendations = recommender.recommend(condition)

                personalized = history_service.get_personalized_recommendations(
                    user_id, condition, base_recommendations
                )

                for drug in personalized:
                    drug_name = drug.get('name', drug.get('drug_name', ''))
                    feedback_stats = history_service.get_drug_feedback_stats(condition, drug_name)
                    if feedback_stats:
                        drug['global_feedback'] = feedback_stats
                    recommendations.append(drug)

            # ── Save consultation (including the user's declared diagnosis) ─
            consultation_id = history_service.save_consultation(
                user_id=user_id,
                symptoms=symptoms,
                condition=condition,
                confidence=result['confidence'],
                recommendations=recommendations,
                user_diagnosis=user_diagnosis          # NEW parameter
            )

            previous_conditions = history_service.get_previous_conditions(user_id)

            checkup_scheduled = None
            if condition in ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']:
                history_service.schedule_checkup(user_id, consultation_id, days_from_now=30)
                checkup_scheduled = '30 days'

            session['prediction'] = {
                'consultation_id': consultation_id,
                'symptoms': symptoms,
                'condition': condition,
                'confidence': result['confidence'],
                'confidence_level': result.get('confidence_level', 'N/A'),
                'probabilities': result.get('probabilities', {}),
                'sentiment': result.get('sentiment', {}),
                'recommendations': recommendations,
                'message': result.get('message', ''),
                'is_ood': result.get('is_ood', False),
                'previous_conditions': previous_conditions,
                'checkup_scheduled': checkup_scheduled,
                # ── NEW diagnosis fields passed to result.html ───────────
                'user_diagnosis': user_diagnosis,
                'diagnosis_agreement': result.get('diagnosis_agreement'),
                'diagnosis_note': result.get('diagnosis_note'),
                'doctor_condition': result.get('doctor_condition'),
                'doctor_condition_prob': result.get('doctor_condition_prob'),
            }

            return render_template('result.html', user=user, **session['prediction'])
        else:
            return render_template(
                'predict.html',
                error=result.get('error', 'Prediction failed'),
                supported_conditions=SUPPORTED_CONDITIONS,
                user=user
            )

    return render_template('predict.html', supported_conditions=SUPPORTED_CONDITIONS, user=user)


@main_bp.route('/dashboard')
@login_required
def dashboard():
    history_service = get_history_service()
    auth_service = get_auth_service()
    user_id = session['user_id']

    user = auth_service.get_user_profile(user_id)
    consultations = history_service.get_user_history(user_id, limit=10)
    checkups = history_service.get_upcoming_checkups(user_id)
    previous_conditions = history_service.get_previous_conditions(user_id)
    stats = history_service.get_user_stats(user_id)

    return render_template('dashboard.html',
                           user=user,
                           consultations=consultations,
                           checkups=checkups,
                           previous_conditions=previous_conditions,
                           stats=stats)


@main_bp.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    history_service = get_history_service()
    user_id = session['user_id']

    consultation_id = request.form.get('consultation_id')
    drug_name = request.form.get('drug_name')
    worked = request.form.get('worked') == 'true'
    effectiveness = request.form.get('effectiveness', type=int)
    side_effects = request.form.get('side_effects', '')
    notes = request.form.get('notes', '')

    if consultation_id and drug_name:
        history_service.save_feedback(
            int(consultation_id), user_id, drug_name, worked,
            effectiveness, side_effects, notes
        )
        return jsonify({'success': True, 'message': 'Thank you for your feedback!'})

    return jsonify({'success': False, 'error': 'Missing required fields'}), 400


@main_bp.route('/history')
@login_required
def history():
    history_service = get_history_service()
    user_id = session['user_id']

    consultations = history_service.get_user_history(user_id)
    return render_template('history.html', consultations=consultations)


@main_bp.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    predictor = get_predictor()
    user_diagnosis = data.get('user_diagnosis')
    result = predictor.predict(data['text'], user_diagnosis=user_diagnosis)

    if result['success']:
        recommender = get_recommender()
        condition = result['condition']
        if condition in recommender.get_all_conditions():
            result['recommendations'] = recommender.recommend(condition)
        else:
            result['recommendations'] = []

    return jsonify(result)


@main_bp.route('/api/learn', methods=['POST'])
def trigger_learning():
    history_service = get_history_service()
    learned_stats = history_service.learn_from_feedback()

    import json
    from pathlib import Path
    learned_path = Path('models/learned_effectiveness.json')
    learned_path.parent.mkdir(exist_ok=True)
    with open(learned_path, 'w') as f:
        json.dump(learned_stats, f, indent=2)

    return jsonify({
        'success': True,
        'conditions_updated': len(learned_stats),
        'stats': learned_stats
    })


@main_bp.route('/api/schedule-checkup', methods=['POST'])
@login_required
def api_schedule_checkup():
    data = request.get_json()
    user_id = session['user_id']

    days = int(data.get('days', 30))
    history_service = get_history_service()
    checkup_id = history_service.schedule_checkup(user_id, None, days)

    return jsonify({'success': True, 'checkup_id': checkup_id})


@main_bp.route('/api/complete-checkup', methods=['POST'])
@login_required
def api_complete_checkup():
    """
    Receives the 3-step checkup form data from the dashboard modal.
    Marks the checkup as completed and optionally schedules the next one.
    """
    data = request.get_json()
    user_id = session['user_id']
    history_service = get_history_service()

    checkup_id       = data.get('checkup_id')
    feeling          = data.get('feeling')
    feeling_notes    = data.get('feeling_notes', '')
    severity         = data.get('severity')
    new_symptoms     = data.get('new_symptoms', '')
    med_adherence    = data.get('med_adherence', '')
    stop_reason      = data.get('stop_reason', '')
    extra_notes      = data.get('extra_notes', '')
    next_checkup_days = data.get('next_checkup_days')

    # Build a human-readable notes string to store
    notes_parts = []
    if feeling:
        feeling_labels = {
            'much_better': 'Much better', 'better': 'Better',
            'same': 'About the same', 'worse': 'A bit worse',
            'much_worse': 'Much worse', 'new_symptoms': 'New symptoms'
        }
        notes_parts.append(f"Feeling: {feeling_labels.get(feeling, feeling)}")
    if severity:
        notes_parts.append(f"Severity: {severity}/10")
    if new_symptoms:
        notes_parts.append(f"New symptoms: {new_symptoms}")
    if med_adherence:
        adherence_labels = {
            'yes_all': 'Taking as prescribed', 'yes_most': 'Mostly taking',
            'stopped': 'Stopped medication', 'not_started': 'Not started',
            'no_meds': 'No medication'
        }
        notes_parts.append(f"Medication: {adherence_labels.get(med_adherence, med_adherence)}")
        if stop_reason:
            notes_parts.append(f"Stop reason: {stop_reason}")
    if extra_notes:
        notes_parts.append(f"Notes: {extra_notes}")

    compiled_notes = ' | '.join(notes_parts)

    # Mark this checkup as completed
    if checkup_id:
        result = history_service.complete_checkup(
            int(checkup_id),
            compiled_notes,
            feeling=feeling,
            severity=int(severity) if severity else None,
            medication_adherence=med_adherence or None,
        )
    else:
        result = None

    # Auto-scheduling is handled inside complete_checkup.
    # If the caller also sent an explicit next_checkup_days override, honour it.
    next_id = None
    auto_days = None
    if result:
        next_id, auto_days = result

    if next_checkup_days and not next_id:
        # User requested a specific date in addition to the auto one
        try:
            next_id = history_service.schedule_checkup(
                user_id, None, int(next_checkup_days)
            )
        except Exception:
            pass

    from datetime import datetime, timedelta
    next_date_str = None
    days_used = auto_days or (int(next_checkup_days) if next_checkup_days else None)
    if days_used:
        next_date_str = (datetime.now() + timedelta(days=days_used)).strftime('%d %B %Y')

    return jsonify({
        'success': True,
        'next_checkup_id': next_id,
        'next_checkup_date': next_date_str,
        'next_checkup_days': days_used,
        'message': (
            f"Check-in recorded. Next follow-up scheduled for {next_date_str}."
            if next_date_str else 'Checkup completed successfully'
        )
    })


@main_bp.route('/about')
def about():
    return render_template('about.html')


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN AUTH
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('is_admin'):
        return redirect(url_for('main.admin'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('main.admin'))
        error = 'Invalid credentials.'

    return render_template('admin_login.html', error=error)


@main_bp.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('main.admin_login'))


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN
# ─────────────────────────────────────────────────────────────────────────────

def _get_csv_stats():
    """Read drugsComTrain_raw.csv and return basic dataset stats."""
    csv_path = Path('data/raw/drugsComTrain_raw.csv')
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')

    total = len(df)
    counts = df['condition'].value_counts()

    depression_count = int(counts.get('Depression', 0))
    hbp_count        = int(counts.get('High Blood Pressure', 0))
    diabetes_count   = int(counts.get('Diabetes, Type 2', 0))

    pct = lambda n: round(n / total * 100, 2) if total else 0

    return {
        'total_records':    total,
        'depression_count': depression_count,
        'depression_pct':   pct(depression_count),
        'hbp_count':        hbp_count,
        'hbp_pct':          pct(hbp_count),
        'diabetes_count':   diabetes_count,
        'diabetes_pct':     pct(diabetes_count),
        'mean_rating':      round(float(df['rating'].mean()), 2),
        'rating_std':       round(float(df['rating'].std()), 2),
    }


@main_bp.route('/admin')
@admin_required
def admin():
    csv_stats = _get_csv_stats()

    # Ensemble + baseline accuracy from tuning results
    tuning_stats = {}
    tuning_path = Path('models/tuning_results.json')
    if tuning_path.exists():
        with open(tuning_path) as f:
            t = json.load(f)
        tuning_stats = {
            'ensemble_accuracy':  round(t.get('test_accuracy', 0) * 100, 2),
            'baseline_accuracy':  round(t.get('baseline_accuracy', 0) * 100, 2),
        }

    # Cached individual model accuracies and per-class F1
    model_stats = {}
    admin_cache = Path('models/admin_stats.json')
    if admin_cache.exists():
        with open(admin_cache) as f:
            model_stats = json.load(f)

    return render_template('admin.html',
                           user=None,
                           csv_stats=csv_stats,
                           tuning_stats=tuning_stats,
                           model_stats=model_stats)


@main_bp.route('/admin/compute-stats', methods=['POST'])
@admin_required
def admin_compute_stats():
    """Compute individual model accuracies and per-class F1, then cache."""
    try:
        import pickle
        import joblib
        from sklearn.metrics import accuracy_score, f1_score

        with open('models/tuned_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        label_encoder = joblib.load('models/label_encoder.pkl')

        df_test = pd.read_csv('data/processed/cleaned_test_data.csv')
        X_test  = df_test['review'].fillna('').values
        y_test  = label_encoder.transform(df_test['condition'].values)

        # Transform once through feature + scaler steps
        X_feat   = pipeline.named_steps['features'].transform(X_test)
        X_scaled = pipeline.named_steps['scaler'].transform(X_feat)

        voting = pipeline.named_steps['classifier']
        lr_acc  = float(accuracy_score(y_test, voting.named_estimators_['lr'].predict(X_scaled)))
        rf_acc  = float(accuracy_score(y_test, voting.named_estimators_['rf'].predict(X_scaled)))
        xgb_acc = float(accuracy_score(y_test, voting.named_estimators_['xgb'].predict(X_scaled)))

        # Ensemble per-class F1
        y_pred = pipeline.predict(X_test)
        classes = label_encoder.classes_
        f1_vals = f1_score(y_test, y_pred, average=None)
        per_class_f1 = {cls: round(float(f), 4) for cls, f in zip(classes, f1_vals)}

        stats = {
            'lr_accuracy':   round(lr_acc * 100, 2),
            'rf_accuracy':   round(rf_acc * 100, 2),
            'xgb_accuracy':  round(xgb_acc * 100, 2),
            'per_class_f1':  per_class_f1,
            'computed_at':   datetime.now().isoformat(),
        }

        with open('models/admin_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return jsonify({'success': True, 'stats': stats})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@main_bp.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404