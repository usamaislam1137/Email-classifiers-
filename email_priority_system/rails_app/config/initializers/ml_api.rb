# ML API configuration
# Override with environment variables in production:
#   ML_API_URL=http://ml-service:5000 bundle exec rails server

ML_API_URL     = ENV.fetch("ML_API_URL",     "http://localhost:5000").freeze
ML_API_TIMEOUT = ENV.fetch("ML_API_TIMEOUT", 30).to_i.freeze
