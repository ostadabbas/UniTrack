# UniTrack: Enhanced Multi-Object Tracking with Unitrack Loss
## GitHub Pages Showcase Website

This repository contains the source code for the UniTrack project's showcase website, designed to demonstrate the effectiveness of our novel Unitrack Loss mechanism in multi-object tracking.

## 🚀 Live Demo

Visit the live website at: [https://yourusername.github.io/UT-MOTR](https://yourusername.github.io/UT-MOTR)

## 📁 Project Structure

```
UT-MOTR/
├── index.html              # Main website page
├── styles.css              # CSS styling
├── script.js               # JavaScript functionality
├── README.md               # This file
├── unitrack_assets/        # Images and logos
│   ├── ut_logo.png
│   ├── unitrack_method.png
│   ├── unitrack_fig3_redraw.png
│   └── fps_ablation.png
├── comparison_output/      # Inference results and videos
│   ├── unitrack_inference/
│   ├── baseline_inference/
│   └── comparison_videos/
│       ├── MOT17-02-FRCNN_comparison.mp4
│       ├── MOT17-04-FRCNN_comparison.mp4
│       ├── MOT17-05-FRCNN_comparison.mp4
│       └── MOT17-09-FRCNN_comparison.mp4

```

## 🎯 Features

### Interactive Video Comparisons
- **Side-by-side comparisons**: Baseline GTR vs UT-GTR performance
- **Video selector**: Switch between different MOT17 sequences
- **Interactive controls**: Play/pause, highlight improvements, show trajectories
- **Real-time switching**: Seamless video transitions

### Professional Design
- **CVPR-style layout**: Academic conference presentation format
- **Responsive design**: Works on desktop, tablet, and mobile devices
- **Smooth animations**: Scroll-based animations and hover effects
- **Modern UI**: Clean, professional interface

### Comprehensive Content
- **Method overview**: Detailed explanation of UniTrack loss components
- **Results analysis**: Performance metrics across multiple datasets
- **Qualitative analysis**: Visual improvements and use cases
- **Download section**: Paper, code, models, and citation

## 🛠️ Deployment Instructions

### Option 1: GitHub Pages (Recommended)

1. **Fork or clone this repository**
2. **Enable GitHub Pages**:
   - Go to repository Settings
   - Scroll to "Pages" section
   - Source: Deploy from a branch
   - Branch: `main` (or your default branch)
   - Folder: `/ (root)`
   - Click "Save"

3. **Wait for deployment** (usually 1-2 minutes)
4. **Access your site** at `https://yourusername.github.io/repository-name`

### Option 2: Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/UT-MOTR.git
   cd UT-MOTR
   ```

2. **Serve locally**:
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Using Python 2
   python -m SimpleHTTPServer 8000
   
   # Using Node.js
   npx http-server .
   ```

3. **Open in browser**: Navigate to `http://localhost:8000`

## 📊 Content Management

### Adding New Videos
1. Place comparison videos in `comparison_output/comparison_videos/`
2. Update the video selector buttons in `index.html`
3. Add corresponding data attributes to selector buttons

### Updating Results
1. Edit the results tables in the Results section of `index.html`
2. Update performance metrics and analysis text
3. Add new dataset tabs as needed

### Customizing Styling
1. Edit `styles.css` for visual changes
2. Modify color scheme by updating CSS variables
3. Adjust responsive breakpoints for mobile optimization

## 🎨 Customization Guide

### Color Scheme
The website uses a professional blue-gray color palette:
- Primary: `#2563eb` (Blue)
- Secondary: `#1e293b` (Dark Gray)
- Accent: `#059669` (Green)
- Background: `#f8fafc` (Light Gray)

### Typography
- Main font: Inter (Google Fonts)
- Monospace: 'Courier New' (for code citations)
- Font sizes: Responsive scaling with rem units

### Layout
- Max width: 1200px
- Grid-based layout with CSS Grid and Flexbox
- Mobile-first responsive design
- Consistent spacing using rem units

## 🔧 Technical Details

### Dependencies
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with Grid and Flexbox
- **Vanilla JavaScript**: No external frameworks required
- **Font Awesome**: Icons (loaded via CDN)
- **Google Fonts**: Inter font family

### Browser Support
- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+
- Mobile browsers (iOS Safari, Chrome Mobile)

### Performance Optimizations
- Lazy loading for images
- Debounced scroll events
- Efficient video loading
- Optimized CSS animations
- Minimal JavaScript footprint

## 📱 Mobile Responsiveness

The website is fully responsive and includes:
- Adaptive grid layouts
- Touch-friendly video controls
- Collapsible navigation
- Optimized font sizes
- Mobile-specific interactions

## 🚨 Troubleshooting

### Videos Not Loading
- Ensure video files are in the correct directory
- Check video file formats (MP4 recommended)
- Verify video paths in HTML and JavaScript

### Styling Issues
- Clear browser cache
- Check CSS file loading
- Verify responsive breakpoints
- Test on different screen sizes

### JavaScript Errors
- Check browser console for error messages
- Ensure all DOM elements exist before accessing
- Verify event listeners are properly attached

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or suggestions regarding the website:
- GitHub Issues: [Create an issue](https://github.com/yourusername/UT-MOTR/issues)
- Email: your.email@example.com

## 🎉 Acknowledgments

- **GTR Framework**: Original Global Tracking Transformer implementation
- **MOT Challenge**: Multi-object tracking benchmark datasets
- **CVPR Community**: Inspiration for academic presentation format
- **Open Source**: Thanks to all contributors and maintainers

---

**Note**: This website showcases the UniTrack research project. For the actual research code and implementation, please refer to the main repository sections.
