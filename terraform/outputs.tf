output "cloud_run_url" {
  description = "Public URL of the deployed bot"
  value       = google_cloud_run_v2_service.bot.uri
}

output "artifact_registry_repo" {
  description = "Artifact Registry repo path for Docker images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/amazon-device-support-bot/app"
}
