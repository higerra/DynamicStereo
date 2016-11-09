uniform sampler2D tex_sampler;

varying vec2 frag_texcoord;
varying float frag_weight;

void main(){
    gl_FragColor = texture2D(tex_sampler, frag_texcoord);
    //vec4 texcolor = texture2D(tex_sampler, frag_texcoord);
    //gl_FragColor = vec4(texcolor.rgb, frag_weight);
}
