# ML API configuration
# Override with environment variables in production:
#   ML_API_URL=http://ml-service:5000 bundle exec rails server
#
# Default port 5001: on macOS, port 5000 is often taken by AirPlay Receiver (ControlCenter).
# Run Flask as: python flask_api.py --port 5001   or fix FLASK_PORT in ml/config.py

ML_API_URL     = ENV.fetch("ML_API_URL",     "http://localhost:5001").freeze
ML_API_TIMEOUT = ENV.fetch("ML_API_TIMEOUT", 30).to_i.freeze
