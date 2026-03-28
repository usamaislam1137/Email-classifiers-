class EmailClassification < ApplicationRecord
  PRIORITIES = %w[critical high normal low].freeze

  PRIORITY_COLORS = {
    "critical" => "danger",
    "high"     => "warning",
    "normal"   => "primary",
    "low"      => "secondary"
  }.freeze

  PRIORITY_ICONS = {
    "critical" => "Red",
    "high"     => "Yellow",
    "normal"   => "Blue",
    "low"      => ""
  }.freeze

  PRIORITY_BG_CLASSES = {
    "critical" => "bg-danger bg-opacity-10 border-danger",
    "high"     => "bg-warning bg-opacity-10 border-warning",
    "normal"   => "bg-primary bg-opacity-10 border-primary",
    "low"      => "bg-secondary bg-opacity-10 border-secondary"
  }.freeze

  # -- Validations -------------------------------------------------------------
  validates :sender,  presence: true
  validates :subject, presence: true
  validates :body,    presence: true
  validates :priority, inclusion: { in: PRIORITIES }, allow_nil: true

  # -- Scopes -------------------------------------------------------------------
  scope :critical, -> { where(priority: "critical") }
  scope :high,     -> { where(priority: "high") }
  scope :normal,   -> { where(priority: "normal") }
  scope :low,      -> { where(priority: "low") }
  scope :recent,   -> { order(created_at: :desc) }

  # -- Helpers ------------------------------------------------------------------

  def priority_color
    PRIORITY_COLORS[priority] || "secondary"
  end

  def priority_icon
    PRIORITY_ICONS[priority] || ""
  end

  def priority_bg_class
    PRIORITY_BG_CLASSES[priority] || "bg-secondary bg-opacity-10"
  end

  def confidence_percent
    return 0 unless confidence
    (confidence * 100).round(1)
  end

  def shap_data
    JSON.parse(shap_values || "{}") rescue {}
  end

  def confidence_data
    JSON.parse(confidence_scores || "{}") rescue {}
  end

  def body_snippet(length = 200)
    return "" if body.blank?
    body.truncate(length)
  end

  def sender_domain
    return "" if sender.blank?
    sender.split("@").last || ""
  end

  def self.priority_distribution
    group(:priority).count.transform_values do |count|
      (count.to_f / self.count * 100).round(1)
    end
  end
end
