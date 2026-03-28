require "httparty"
require "json"

# MlApiClient
#
# Service object that wraps all communication with the Python Flask ML API.
# Falls back gracefully if the API is unavailable.
#
class MlApiClient
  include HTTParty

  HEADERS = { "Content-Type" => "application/json", "Accept" => "application/json" }.freeze

  # -- classify --------------------------------------------------------------
  #
  # Sends a POST /predict request to the ML API.
  #
  # Returns a hash with:
  #   success:           true | false
  #   priority:          "critical" | "high" | "normal" | "low"
  #   confidence:        Float (0..1)
  #   confidence_scores: Hash {class => score}
  #   shap_values:       Hash {feature => importance}
  #   model_used:        String
  #   processing_time_ms: Integer
  #   error:             String (only when success: false)
  #
  def self.classify(sender:, recipients: nil, subject:, body:, date: nil, cc: nil)
    payload = {
      sender:     sender,
      recipients: recipients,
      subject:    subject,
      body:       body,
      date:       date || Time.current.iso8601,
      cc:         cc
    }.compact.to_json

    response = HTTParty.post(
      "#{ML_API_URL}/predict",
      body:    payload,
      headers: HEADERS,
      timeout: ML_API_TIMEOUT
    )

    if response.success?
      data = JSON.parse(response.body, symbolize_names: true)
      { success: true }.merge(data)
    else
      body_text = response.body.to_s.truncate(300)
      {
        success: false,
        error:   "ML API returned HTTP #{response.code}: #{body_text}"
      }
    end
  rescue HTTParty::Error, SocketError, Errno::ECONNREFUSED => e
    {
      success: false,
      error:   "ML API connection failed: #{e.message}"
    }
  rescue Net::OpenTimeout, Net::ReadTimeout => e
    {
      success: false,
      error:   "ML API timed out after #{ML_API_TIMEOUT}s: #{e.message}"
    }
  rescue JSON::ParserError => e
    {
      success: false,
      error:   "ML API returned invalid JSON: #{e.message}"
    }
  rescue StandardError => e
    {
      success: false,
      error:   "ML API unexpected error: #{e.message}"
    }
  end

  # -- batch_classify --------------------------------------------------------
  #
  # Sends a POST /batch_predict request.
  # Returns hash with results: Array and total: Integer.
  #
  def self.batch_classify(emails:)
    payload = { emails: emails }.to_json

    response = HTTParty.post(
      "#{ML_API_URL}/batch_predict",
      body:    payload,
      headers: HEADERS,
      timeout: ML_API_TIMEOUT * 3   # larger timeout for batches
    )

    if response.success?
      JSON.parse(response.body, symbolize_names: true)
    else
      { success: false, error: "Batch predict failed (HTTP #{response.code})" }
    end
  rescue StandardError => e
    { success: false, error: e.message }
  end

  # -- model_info ------------------------------------------------------------
  #
  # Returns info about the currently active model, or nil on failure.
  #
  def self.model_info
    response = HTTParty.get(
      "#{ML_API_URL}/model_info",
      headers: HEADERS,
      timeout: 5
    )
    JSON.parse(response.body, symbolize_names: true) if response.success?
  rescue StandardError
    nil
  end

  # -- health_check ----------------------------------------------------------
  #
  # Returns true if the ML API is reachable and healthy.
  #
  def self.health_check
    response = HTTParty.get(
      "#{ML_API_URL}/health",
      headers: HEADERS,
      timeout: 3
    )
    response.success? && JSON.parse(response.body)["status"] == "ok"
  rescue StandardError
    false
  end

  # -- list_models -----------------------------------------------------------
  def self.list_models
    response = HTTParty.get(
      "#{ML_API_URL}/models",
      headers: HEADERS,
      timeout: 5
    )
    JSON.parse(response.body, symbolize_names: true) if response.success?
  rescue StandardError
    nil
  end
end
