class CreateEmailClassifications < ActiveRecord::Migration[7.2]
  def change
    create_table :email_classifications do |t|
      t.string  :sender,             null: false
      t.string  :recipients
      t.string  :subject,            null: false
      t.text    :body,               null: false
      t.string  :date
      t.string  :priority
      t.float   :confidence
      t.text    :confidence_scores
      t.text    :shap_values
      t.string  :model_used
      t.integer :processing_time_ms

      t.timestamps
    end

    add_index :email_classifications, :priority
    add_index :email_classifications, :created_at
    add_index :email_classifications, :sender
  end
end
