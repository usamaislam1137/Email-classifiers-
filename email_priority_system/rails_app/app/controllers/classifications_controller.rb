class ClassificationsController < ApplicationController
  before_action :set_classification, only: [:show, :destroy]

  def index
    @classifications = EmailClassification.order(created_at: :desc).limit(50)
    @priority_counts = EmailClassification.group(:priority).count
  end

  def new
    @classification = EmailClassification.new
    # Pre-fill with a sample email for demo purposes
    @sample_emails = [
      {
        sender: "ceo@company.com",
        subject: "URGENT: Board meeting tomorrow - action required",
        body: "Please respond immediately. This is time sensitive and requires your attention ASAP. The board needs a decision by EOD today."
      },
      {
        sender: "colleague@company.com",
        subject: "Follow up: Q4 report deadline by Friday",
        body: "Hi, just following up on the Q4 report. Please ensure the draft is ready by Friday EOD. Let me know if you need anything."
      },
      {
        sender: "newsletter@marketing.com",
        subject: "Your weekly digest is here!",
        body: "Here is your weekly newsletter. Unsubscribe at any time. This is an automated message sent to all subscribers."
      }
    ]
  end

  def create
    @classification = EmailClassification.new(classification_params)

    # Call the ML API
    result = MlApiClient.classify(
      sender: params[:email_classification][:sender],
      recipients: params[:email_classification][:recipients],
      subject: params[:email_classification][:subject],
      body: params[:email_classification][:body],
      date: params[:email_classification][:date]
    )

    if result[:success]
      @classification.priority             = result[:priority]
      @classification.confidence           = result[:confidence]
      @classification.confidence_scores    = result[:confidence_scores].to_json
      @classification.shap_values          = result[:shap_values].to_json
      @classification.model_used           = result[:model_used]
      @classification.processing_time_ms   = result[:processing_time_ms]

      if @classification.save
        redirect_to classification_path(@classification), notice: "Email classified as #{result[:priority].capitalize}!"
      else
        @error = "Failed to save classification: #{@classification.errors.full_messages.join(', ')}"
        render :new, status: :unprocessable_entity
      end
    else
      @error = result[:error] || "ML API classification failed."
      render :new, status: :unprocessable_entity
    end
  end

  def show
    @shap_values       = JSON.parse(@classification.shap_values || "{}") rescue {}
    @confidence_scores = JSON.parse(@classification.confidence_scores || "{}") rescue {}

    # Sort SHAP values by importance descending, keep top 10
    @shap_sorted = @shap_values.sort_by { |_, v| -v.to_f }.first(10).to_h
  end

  def destroy
    @classification.destroy
    redirect_to classifications_path, notice: "Classification record deleted."
  end

  private

  def set_classification
    @classification = EmailClassification.find(params[:id])
  end

  def classification_params
    params.require(:email_classification).permit(
      :sender, :recipients, :subject, :body, :date
    )
  end
end
