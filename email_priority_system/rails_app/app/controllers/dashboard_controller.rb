class DashboardController < ApplicationController
  def index
    @total           = EmailClassification.count
    @priority_counts = EmailClassification.group(:priority).count
    @recent          = EmailClassification.order(created_at: :desc).limit(10)

    # Per-priority counts for stat cards
    @critical_count = @priority_counts["critical"] || 0
    @high_count     = @priority_counts["high"]     || 0
    @normal_count   = @priority_counts["normal"]   || 0
    @low_count      = @priority_counts["low"]      || 0

    # Average confidence
    @avg_confidence = EmailClassification.average(:confidence)&.round(3) || 0

    # Fetch ML API info (non-blocking: errors are caught)
    @model_info  = begin
      MlApiClient.model_info
    rescue StandardError
      nil
    end

    @api_healthy = MlApiClient.health_check
  end

  def health
    healthy = MlApiClient.health_check
    render json: {
      status:  healthy ? "ok" : "degraded",
      ml_api:  healthy,
      db:      ActiveRecord::Base.connection.active?,
      version: Rails.version
    }
  end
end
