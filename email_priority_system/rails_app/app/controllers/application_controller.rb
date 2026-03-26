class ApplicationController < ActionController::Base
  # Protect from CSRF attacks
  protect_from_forgery with: :exception

  # -- Common error handlers --------------------------------------------------
  rescue_from ActiveRecord::RecordNotFound do |e|
    respond_to do |format|
      format.html { render "errors/not_found", status: :not_found }
      format.json { render json: { error: "Record not found", message: e.message }, status: :not_found }
    end
  end

  rescue_from ActionController::ParameterMissing do |e|
    respond_to do |format|
      format.html do
        flash[:alert] = "Missing required parameter: #{e.param}"
        redirect_back fallback_location: root_path
      end
      format.json { render json: { error: "Missing parameter", message: e.message }, status: :bad_request }
    end
  end

  rescue_from Net::OpenTimeout, Net::ReadTimeout do |_e|
    respond_to do |format|
      format.html do
        flash[:alert] = "The ML API request timed out. Please try again."
        redirect_back fallback_location: root_path
      end
      format.json { render json: { error: "Request timeout" }, status: :gateway_timeout }
    end
  end
end
