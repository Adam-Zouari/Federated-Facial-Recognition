package com.faceid

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.faceid.data.FaceEmbedding

class FaceListAdapter(
    private val onItemClick: (FaceEmbedding) -> Unit
) : ListAdapter<FaceEmbedding, FaceListAdapter.FaceViewHolder>(FaceDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): FaceViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_face, parent, false)
        return FaceViewHolder(view, onItemClick)
    }

    override fun onBindViewHolder(holder: FaceViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    class FaceViewHolder(
        itemView: View,
        private val onItemClick: (FaceEmbedding) -> Unit
    ) : RecyclerView.ViewHolder(itemView) {
        
        private val tvName: TextView = itemView.findViewById(R.id.tv_name)
        private val tvDetails: TextView = itemView.findViewById(R.id.tv_details)

        fun bind(face: FaceEmbedding) {
            tvName.text = "👤 ${face.name}"
            tvDetails.text = "Poses: ${face.numPoses} • ${face.dateAdded}"
            
            itemView.setOnClickListener {
                onItemClick(face)
            }
        }
    }

    private class FaceDiffCallback : DiffUtil.ItemCallback<FaceEmbedding>() {
        override fun areItemsTheSame(oldItem: FaceEmbedding, newItem: FaceEmbedding): Boolean {
            return oldItem.name == newItem.name
        }

        override fun areContentsTheSame(oldItem: FaceEmbedding, newItem: FaceEmbedding): Boolean {
            return oldItem == newItem
        }
    }
}
